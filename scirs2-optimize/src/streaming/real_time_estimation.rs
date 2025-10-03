//! Real-time Parameter Estimation
//!
//! This module provides algorithms for real-time parameter estimation that can
//! handle continuous data streams with minimal latency. These methods are designed
//! for applications where immediate response to new data is critical.

use super::{
    utils, StreamingConfig, StreamingDataPoint, StreamingObjective, StreamingOptimizer,
    StreamingStats,
};
use crate::error::OptimizeError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
// Unused import
// use scirs2_core::error::CoreResult;
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, OptimizeError>;

/// Real-time estimation methods
#[derive(Debug, Clone, Copy)]
pub enum RealTimeMethod {
    /// Recursive Least Squares (RLS)
    RecursiveLeastSquares,
    /// Kalman Filter for parameter estimation
    KalmanFilter,
    /// Exponentially Weighted RLS
    ExponentiallyWeightedRLS,
    /// Sliding Window RLS
    SlidingWindowRLS,
}

/// Real-time parameter estimator
#[derive(Debug, Clone)]
pub struct RealTimeEstimator<T: StreamingObjective> {
    /// Current parameter estimates
    parameters: Array1<f64>,
    /// Objective function
    objective: T,
    /// Configuration
    config: StreamingConfig,
    /// Statistics
    stats: StreamingStats,
    /// Estimation method
    method: RealTimeMethod,
    /// Covariance matrix (for RLS and Kalman filter)
    covariance: Array2<f64>,
    /// Forgetting factor for exponential weighting
    forgetting_factor: f64,
    /// Process noise for Kalman filter
    process_noise: f64,
    /// Measurement noise for Kalman filter
    measurement_noise: f64,
    /// Time of last update for adaptive algorithms
    last_update_time: Option<Instant>,
    /// Sliding window for windowed methods
    window_data: std::collections::VecDeque<(Array1<f64>, f64)>, // (features, target)
    /// Maximum processing time per update (for real-time constraints)
    max_processing_time: Duration,
}

impl<T: StreamingObjective> RealTimeEstimator<T> {
    /// Create a new real-time estimator
    pub fn new(
        initial_parameters: Array1<f64>,
        objective: T,
        config: StreamingConfig,
        method: RealTimeMethod,
        initial_covariance_scale: f64,
    ) -> Self {
        let n_params = initial_parameters.len();
        let initial_covariance = Array2::eye(n_params) * initial_covariance_scale;
        let forgetting_factor = config.forgetting_factor;
        let window_size = config.window_size;

        Self {
            parameters: initial_parameters,
            objective,
            config,
            stats: StreamingStats::default(),
            method,
            covariance: initial_covariance,
            forgetting_factor,
            process_noise: 1e-6,
            measurement_noise: 1e-3,
            last_update_time: None,
            window_data: std::collections::VecDeque::with_capacity(window_size),
            max_processing_time: Duration::from_millis(10), // 10ms max for real-time
        }
    }

    /// Update using Recursive Least Squares
    fn update_rls(&mut self, features: &ArrayView1<f64>, target: f64) -> Result<()> {
        let n = features.len();

        // Compute prediction error
        let prediction = self.parameters.dot(features);
        let error = target - prediction;

        // Compute gain vector: K = P * x / (1 + x^T * P * x)
        let mut px = Array1::zeros(n);
        for i in 0..n {
            for j in 0..n {
                px[i] += self.covariance[[i, j]] * features[j];
            }
        }

        let denominator = 1.0 + features.dot(&px);
        if denominator.abs() < 1e-12 {
            return Ok(()); // Skip update if denominator too small
        }

        let gain = &px / denominator;

        // Update parameters: θ = θ + K * error
        self.parameters = &self.parameters + &(error * &gain);

        // Update covariance: P = P - K * x^T * P
        for i in 0..n {
            for j in 0..n {
                self.covariance[[i, j]] -= gain[i] * px[j];
            }
        }

        Ok(())
    }

    /// Update using Exponentially Weighted RLS
    fn update_ewrls(&mut self, features: &ArrayView1<f64>, target: f64) -> Result<()> {
        let n = features.len();

        // Scale covariance by forgetting factor
        self.covariance *= 1.0 / self.forgetting_factor;

        // Compute prediction error
        let prediction = self.parameters.dot(features);
        let error = target - prediction;

        // Compute gain vector: K = P * x / (λ + x^T * P * x)
        let mut px = Array1::zeros(n);
        for i in 0..n {
            for j in 0..n {
                px[i] += self.covariance[[i, j]] * features[j];
            }
        }

        let denominator = self.forgetting_factor + features.dot(&px);
        if denominator.abs() < 1e-12 {
            return Ok(());
        }

        let gain = &px / denominator;

        // Update parameters
        self.parameters = &self.parameters + &(error * &gain);

        // Update covariance
        for i in 0..n {
            for j in 0..n {
                self.covariance[[i, j]] -= gain[i] * px[j];
            }
        }

        Ok(())
    }

    /// Update using Kalman Filter
    fn update_kalman(&mut self, features: &ArrayView1<f64>, target: f64) -> Result<()> {
        let n = features.len();

        // Time update (prediction step)
        // Add process noise to covariance
        for i in 0..n {
            self.covariance[[i, i]] += self.process_noise;
        }

        // Measurement update (correction step)
        let prediction = self.parameters.dot(features);
        let innovation = target - prediction;

        // Innovation covariance: S = H * P * H^T + R
        // For linear case: H = x^T, so S = x^T * P * x + R
        let mut px = Array1::zeros(n);
        for i in 0..n {
            for j in 0..n {
                px[i] += self.covariance[[i, j]] * features[j];
            }
        }

        let innovation_covariance = features.dot(&px) + self.measurement_noise;
        if innovation_covariance.abs() < 1e-12 {
            return Ok(());
        }

        // Kalman gain: K = P * H^T / S = P * x / S
        let kalman_gain = &px / innovation_covariance;

        // Update parameters: θ = θ + K * innovation
        self.parameters = &self.parameters + &(innovation * &kalman_gain);

        // Update covariance: P = (I - K * H) * P = P - K * x^T * P
        for i in 0..n {
            for j in 0..n {
                self.covariance[[i, j]] -= kalman_gain[i] * px[j];
            }
        }

        Ok(())
    }

    /// Update using Sliding Window RLS
    fn update_sliding_window_rls(&mut self, features: &ArrayView1<f64>, target: f64) -> Result<()> {
        // Add new data point
        self.window_data.push_back((features.to_owned(), target));

        // Remove old data if window is full
        if self.window_data.len() > self.config.window_size {
            self.window_data.pop_front();
        }

        // Rebuild normal equations from window
        let n = features.len();
        let mut xtx = Array2::zeros((n, n));
        let mut xty = Array1::zeros(n);

        for (x, y) in &self.window_data {
            for i in 0..n {
                for j in 0..n {
                    xtx[[i, j]] += x[i] * x[j];
                }
                xty[i] += x[i] * y;
            }
        }

        // Add regularization
        for i in 0..n {
            xtx[[i, i]] += self.config.regularization;
        }

        // Solve normal equations
        match scirs2_linalg::solve(&xtx.view(), &xty.view(), None) {
            Ok(solution) => {
                self.parameters = solution;
                // Update covariance as pseudo-inverse of X^T X
                match scirs2_linalg::compat::pinv(&xtx.view(), None, false, true) {
                    Ok(pinv) => self.covariance = pinv,
                    Err(_) => {} // Keep old covariance if inversion fails
                }
            }
            Err(_) => {
                // Fall back to RLS update if linear system fails
                self.update_rls(features, target)?;
            }
        }

        Ok(())
    }

    /// Adaptive parameter tuning based on performance
    fn adapt_parameters(&mut self) {
        // Adapt forgetting factor based on recent performance
        if self.stats.points_processed > 10 {
            let recent_loss_trend = self.stats.current_loss - self.stats.average_loss;

            if recent_loss_trend > 0.0 {
                // Performance is getting worse, reduce forgetting factor (adapt faster)
                self.forgetting_factor = (self.forgetting_factor * 0.95).max(0.5);
            } else {
                // Performance is stable, increase forgetting factor (adapt slower)
                self.forgetting_factor = (self.forgetting_factor * 1.01).min(0.999);
            }
        }

        // Adapt noise parameters for Kalman filter
        if matches!(self.method, RealTimeMethod::KalmanFilter) {
            let param_change_rate = if self.stats.points_processed > 1 {
                // Estimate parameter change rate from recent updates
                // This is simplified - in practice, track parameter history
                self.parameters.mapv(|x| x.abs()).sum() / self.stats.points_processed as f64
            } else {
                1e-6
            };

            self.process_noise = (param_change_rate * 0.1).max(1e-8).min(1e-3);
        }
    }

    /// Check if update should be skipped due to time constraints
    fn should_skip_for_timing(&self, starttime: Instant) -> bool {
        starttime.elapsed() > self.max_processing_time
    }
}

impl<T: StreamingObjective + Clone> StreamingOptimizer for RealTimeEstimator<T> {
    fn update(&mut self, datapoint: &StreamingDataPoint) -> Result<()> {
        let start_time = Instant::now();

        // Skip update if timing constraints are violated
        if self.should_skip_for_timing(start_time) {
            return Ok(());
        }

        let old_parameters = self.parameters.clone();
        let features = &datapoint.features;
        let target = datapoint.target;

        // Apply method-specific update
        match self.method {
            RealTimeMethod::RecursiveLeastSquares => {
                self.update_rls(&features.view(), target)?;
            }
            RealTimeMethod::ExponentiallyWeightedRLS => {
                self.update_ewrls(&features.view(), target)?;
            }
            RealTimeMethod::KalmanFilter => {
                self.update_kalman(&features.view(), target)?;
            }
            RealTimeMethod::SlidingWindowRLS => {
                self.update_sliding_window_rls(&features.view(), target)?;
            }
        }

        // Adaptive parameter tuning
        if self.stats.points_processed.is_multiple_of(20) {
            self.adapt_parameters();
        }

        // Update statistics
        let loss = self.objective.evaluate(&self.parameters.view(), datapoint);
        self.stats.points_processed += 1;
        self.stats.updates_performed += 1;
        self.stats.current_loss = loss;
        self.stats.average_loss = utils::ewma_update(
            self.stats.average_loss,
            loss,
            0.05, // Faster adaptation for real-time
        );

        // Check convergence
        self.stats.converged = utils::check_convergence(
            &old_parameters.view(),
            &self.parameters.view(),
            self.config.tolerance,
        );

        self.stats.processing_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        self.last_update_time = Some(start_time);

        Ok(())
    }

    fn parameters(&self) -> &Array1<f64> {
        &self.parameters
    }

    fn stats(&self) -> &StreamingStats {
        &self.stats
    }

    fn reset(&mut self) {
        let n = self.parameters.len();
        self.covariance = Array2::eye(n) * 1000.0; // Reset with high uncertainty
        self.forgetting_factor = self.config.forgetting_factor;
        self.last_update_time = None;
        self.window_data.clear();
        self.stats = StreamingStats::default();
    }
}

/// Create a Recursive Least Squares estimator
#[allow(dead_code)]
pub fn recursive_least_squares<T: StreamingObjective>(
    initial_parameters: Array1<f64>,
    objective: T,
    config: Option<StreamingConfig>,
    initial_uncertainty: Option<f64>,
) -> RealTimeEstimator<T> {
    let config = config.unwrap_or_default();
    let uncertainty = initial_uncertainty.unwrap_or(1000.0);

    RealTimeEstimator::new(
        initial_parameters,
        objective,
        config,
        RealTimeMethod::RecursiveLeastSquares,
        uncertainty,
    )
}

/// Create an Exponentially Weighted RLS estimator
#[allow(dead_code)]
pub fn exponentially_weighted_rls<T: StreamingObjective>(
    initial_parameters: Array1<f64>,
    objective: T,
    config: Option<StreamingConfig>,
    forgetting_factor: Option<f64>,
) -> RealTimeEstimator<T> {
    let mut config = config.unwrap_or_default();
    if let Some(ff) = forgetting_factor {
        config.forgetting_factor = ff;
    }

    RealTimeEstimator::new(
        initial_parameters,
        objective,
        config,
        RealTimeMethod::ExponentiallyWeightedRLS,
        100.0,
    )
}

/// Create a Kalman Filter estimator
#[allow(dead_code)]
pub fn kalman_filter_estimator<T: StreamingObjective>(
    initial_parameters: Array1<f64>,
    objective: T,
    config: Option<StreamingConfig>,
    process_noise: Option<f64>,
    measurement_noise: Option<f64>,
) -> RealTimeEstimator<T> {
    let config = config.unwrap_or_default();
    let mut estimator = RealTimeEstimator::new(
        initial_parameters,
        objective,
        config,
        RealTimeMethod::KalmanFilter,
        1.0,
    );

    if let Some(pn) = process_noise {
        estimator.process_noise = pn;
    }
    if let Some(mn) = measurement_noise {
        estimator.measurement_noise = mn;
    }

    estimator
}

/// Convenience function for real-time linear regression
#[allow(dead_code)]
pub fn real_time_linear_regression(
    n_features: usize,
    method: RealTimeMethod,
    config: Option<StreamingConfig>,
) -> RealTimeEstimator<super::LinearRegressionObjective> {
    let config = config.unwrap_or_default();
    let initial_params = Array1::zeros(n_features);
    let objective = super::LinearRegressionObjective;

    RealTimeEstimator::new(initial_params, objective, config, method, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{LinearRegressionObjective, StreamingDataPoint};

    #[test]
    fn test_rls_creation() {
        let estimator =
            recursive_least_squares(Array1::zeros(2), LinearRegressionObjective, None, None);

        assert_eq!(estimator.parameters().len(), 2);
        assert!(matches!(
            estimator.method,
            RealTimeMethod::RecursiveLeastSquares
        ));
    }

    #[test]
    fn test_rls_update() {
        let mut estimator =
            real_time_linear_regression(2, RealTimeMethod::RecursiveLeastSquares, None);

        let features = Array1::from(vec![1.0, 2.0]);
        let target = 3.0;
        let point = StreamingDataPoint::new(features, target);

        assert!(estimator.update(&point).is_ok());
        assert_eq!(estimator.stats().points_processed, 1);
    }

    #[test]
    fn test_ewrls_adaptation() {
        let mut config = StreamingConfig::default();
        config.forgetting_factor = 0.9;

        let mut estimator = exponentially_weighted_rls(
            Array1::zeros(2),
            LinearRegressionObjective,
            Some(config),
            None,
        );

        // Process several data points
        for i in 0..10 {
            let features = Array1::from(vec![i as f64, 1.0]);
            let target = 2.0 * i as f64 + 1.0;
            let point = StreamingDataPoint::new(features, target);

            estimator.update(&point).unwrap();
        }

        assert_eq!(estimator.stats().points_processed, 10);
        assert!(estimator.stats().current_loss.is_finite());
    }

    #[test]
    fn test_kalman_filter() {
        let mut estimator = kalman_filter_estimator(
            Array1::zeros(2),
            LinearRegressionObjective,
            None,
            Some(1e-6),
            Some(1e-3),
        );

        // Add noisy data
        let data_points = vec![
            StreamingDataPoint::new(Array1::from(vec![1.0, 0.0]), 2.1),
            StreamingDataPoint::new(Array1::from(vec![0.0, 1.0]), 2.9),
            StreamingDataPoint::new(Array1::from(vec![1.0, 1.0]), 5.1),
        ];

        for point in &data_points {
            estimator.update(point).unwrap();
        }

        assert_eq!(estimator.stats().points_processed, 3);

        // Parameters should be close to [2, 3] despite noise
        let params = estimator.parameters();
        assert!((params[0] - 2.0).abs() < 1.0);
        assert!((params[1] - 3.0).abs() < 1.0);
    }

    #[test]
    fn test_sliding_window_rls() {
        let mut estimator = real_time_linear_regression(2, RealTimeMethod::SlidingWindowRLS, None);

        // Add data points to exceed window size
        for i in 0..15 {
            let features = Array1::from(vec![i as f64, 1.0]);
            let target = 2.0 * i as f64;
            let point = StreamingDataPoint::new(features, target);

            estimator.update(&point).unwrap();
        }

        // Should have processed all points but window is limited
        assert_eq!(estimator.stats().points_processed, 15);
        assert!(estimator.window_data.len() <= estimator.config.window_size);
    }

    #[test]
    fn test_covariance_updates() {
        let mut estimator = recursive_least_squares(
            Array1::zeros(2),
            LinearRegressionObjective,
            None,
            Some(100.0),
        );

        let initial_covariance = estimator.covariance.clone();

        let features = Array1::from(vec![1.0, 1.0]);
        let target = 1.0;
        let point = StreamingDataPoint::new(features, target);

        estimator.update(&point).unwrap();

        // Covariance should change after update
        assert!(&estimator.covariance != &initial_covariance);

        // Diagonal elements should generally decrease (uncertainty reduction)
        assert!(estimator.covariance[[0, 0]] < initial_covariance[[0, 0]]);
        assert!(estimator.covariance[[1, 1]] < initial_covariance[[1, 1]]);
    }

    #[test]
    fn test_real_time_constraints() {
        let mut estimator =
            real_time_linear_regression(2, RealTimeMethod::RecursiveLeastSquares, None);

        // Set very tight timing constraint
        estimator.max_processing_time = Duration::from_nanos(1);

        let features = Array1::from(vec![1.0, 2.0]);
        let target = 3.0;
        let point = StreamingDataPoint::new(features, target);

        // Update should complete quickly (might skip processing due to timing)
        let start = Instant::now();
        estimator.update(&point).unwrap();
        let elapsed = start.elapsed();

        // Should not take more than a reasonable amount of time
        assert!(elapsed < Duration::from_millis(100));
    }

    #[test]
    fn test_parameter_adaptation() {
        let mut estimator = exponentially_weighted_rls(
            Array1::zeros(2),
            LinearRegressionObjective,
            None,
            Some(0.95),
        );

        let initial_ff = estimator.forgetting_factor;

        // Process many points to trigger adaptation
        for i in 0..50 {
            let features = Array1::from(vec![i as f64, 1.0]);
            let target = i as f64; // Potentially changing relationship
            let point = StreamingDataPoint::new(features, target);

            estimator.update(&point).unwrap();
        }

        // Forgetting factor may have been adapted
        // (exact behavior depends on loss trends)
        assert!(estimator.stats().points_processed == 50);
    }
}
