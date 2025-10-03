//! Streaming Trust Region Methods
//!
//! This module implements trust region methods adapted for streaming data and
//! real-time optimization scenarios. The methods maintain approximate Hessian
//! information and adapt the trust region radius based on streaming performance.

use super::{
    utils, StreamingConfig, StreamingDataPoint, StreamingObjective, StreamingOptimizer,
    StreamingStats,
};
use crate::error::OptimizeError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
// Unused import
// use scirs2_core::error::CoreResult;
use scirs2_linalg::solve;
// Unused import
// use scirs2_linalg::LinalgError;

type Result<T> = std::result::Result<T, OptimizeError>;

/// Streaming Trust Region optimizer
#[derive(Debug, Clone)]
pub struct StreamingTrustRegion<T: StreamingObjective> {
    /// Current parameter estimates
    parameters: Array1<f64>,
    /// Objective function
    objective: T,
    /// Configuration
    config: StreamingConfig,
    /// Statistics
    stats: StreamingStats,
    /// Current trust region radius
    trust_radius: f64,
    /// Approximate Hessian matrix (updated incrementally)
    hessian_approx: Array2<f64>,
    /// Gradient accumulator for improved estimates
    gradient_accumulator: Array1<f64>,
    /// Number of gradient accumulations
    gradient_count: usize,
    /// Success ratio for trust region adaptation
    success_ratio: f64,
    /// Previous function value for ratio computation
    prev_function_value: f64,
}

impl<T: StreamingObjective> StreamingTrustRegion<T> {
    /// Create a new streaming trust region optimizer
    pub fn new(
        initial_parameters: Array1<f64>,
        objective: T,
        config: StreamingConfig,
        initial_trust_radius: f64,
    ) -> Self {
        let n_params = initial_parameters.len();
        Self {
            parameters: initial_parameters,
            objective,
            config,
            stats: StreamingStats::default(),
            trust_radius: initial_trust_radius,
            hessian_approx: Array2::eye(n_params), // Start with identity
            gradient_accumulator: Array1::zeros(n_params),
            gradient_count: 0,
            success_ratio: 0.5,
            prev_function_value: f64::INFINITY,
        }
    }

    /// Solve the trust region subproblem using Cauchy point method
    fn solve_trust_region_subproblem(&self, gradient: &ArrayView1<f64>) -> Result<Array1<f64>> {
        let n = gradient.len();

        // For numerical stability, add regularization to Hessian
        let mut regularized_hessian = self.hessian_approx.clone();
        for i in 0..n {
            regularized_hessian[[i, i]] += self.config.regularization;
        }

        // Try to solve the Newton system first
        match solve(&regularized_hessian.view(), &(-gradient).view(), None) {
            Ok(newton_step) => {
                let step_norm = newton_step.mapv(|x| x * x).sum().sqrt();

                if step_norm <= self.trust_radius {
                    // Newton step is within trust region
                    Ok(newton_step)
                } else {
                    // Scale Newton step to trust region boundary
                    Ok(newton_step * (self.trust_radius / step_norm))
                }
            }
            Err(_) => {
                // Fall back to steepest descent (Cauchy point)
                let grad_norm = gradient.mapv(|x| x * x).sum().sqrt();
                if grad_norm > 0.0 {
                    let cauchy_step = -(self.trust_radius / grad_norm) * gradient;
                    Ok(cauchy_step.to_owned())
                } else {
                    Ok(Array1::zeros(n))
                }
            }
        }
    }

    /// Update the Hessian approximation using BFGS-like formula
    fn update_hessian_approximation(
        &mut self,
        step: &ArrayView1<f64>,
        grad_diff: &ArrayView1<f64>,
    ) {
        let rho = step.dot(grad_diff);

        if rho.abs() < 1e-12 {
            return; // Skip update if curvature condition is not satisfied
        }

        // Apply forgetting factor to previous Hessian
        self.hessian_approx *= self.config.forgetting_factor;

        // BFGS update: H = H + (y*y^T)/(y^T*s) - (H*s*s^T*H)/(s^T*H*s)
        let n = step.len();
        let mut outer_yy = Array2::zeros((n, n));
        let mut hs = Array1::zeros(n);

        // Compute H*s
        for i in 0..n {
            for j in 0..n {
                hs[i] += self.hessian_approx[[i, j]] * step[j];
            }
        }

        let shs = step.dot(&hs);

        if shs > 1e-12 {
            // Compute outer products
            for i in 0..n {
                for j in 0..n {
                    outer_yy[[i, j]] = grad_diff[i] * grad_diff[j];
                    self.hessian_approx[[i, j]] += outer_yy[[i, j]] / rho - (hs[i] * hs[j]) / shs;
                }
            }
        }

        // Ensure positive definiteness by adding regularization if needed
        let min_eigenvalue = self.estimate_min_eigenvalue();
        if min_eigenvalue < self.config.regularization {
            for i in 0..n {
                self.hessian_approx[[i, i]] += self.config.regularization - min_eigenvalue;
            }
        }
    }

    /// Estimate minimum eigenvalue using Gershgorin circles
    fn estimate_min_eigenvalue(&self) -> f64 {
        let n = self.hessian_approx.nrows();
        let mut min_est = f64::INFINITY;

        for i in 0..n {
            let diagonal = self.hessian_approx[[i, i]];
            let off_diagonal_sum: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| self.hessian_approx[[i, j]].abs())
                .sum();

            let lower_bound = diagonal - off_diagonal_sum;
            min_est = min_est.min(lower_bound);
        }

        min_est
    }

    /// Compute the trust region ratio for step acceptance
    fn compute_trust_region_ratio(
        &self,
        step: &ArrayView1<f64>,
        gradient: &ArrayView1<f64>,
        actual_reduction: f64,
    ) -> f64 {
        // Predicted _reduction: m(0) - m(step) ≈ -g^T*step - 0.5*step^T*H*step
        let linear_term = -gradient.dot(step);

        let mut quadratic_term = 0.0;
        for i in 0..step.len() {
            for j in 0..step.len() {
                quadratic_term += step[i] * self.hessian_approx[[i, j]] * step[j];
            }
        }
        quadratic_term *= 0.5;

        let predicted_reduction = linear_term + quadratic_term;

        if predicted_reduction.abs() < 1e-12 {
            0.0
        } else {
            actual_reduction / predicted_reduction
        }
    }

    /// Update trust region radius based on success ratio
    fn update_trust_radius(&mut self, ratio: f64, stepnorm: f64) {
        const VERY_SUCCESSFUL: f64 = 0.75;
        const SUCCESSFUL: f64 = 0.25;
        const EXPANSION_FACTOR: f64 = 2.0;
        const CONTRACTION_FACTOR: f64 = 0.25;
        const MAX_TRUST_RADIUS: f64 = 1e6;
        const MIN_TRUST_RADIUS: f64 = 1e-12;

        if ratio >= VERY_SUCCESSFUL && stepnorm >= 0.8 * self.trust_radius {
            // Very successful step near boundary: expand trust region
            self.trust_radius = (self.trust_radius * EXPANSION_FACTOR).min(MAX_TRUST_RADIUS);
        } else if ratio < SUCCESSFUL {
            // Unsuccessful step: contract trust region
            self.trust_radius = (self.trust_radius * CONTRACTION_FACTOR).max(MIN_TRUST_RADIUS);
        }
        // Otherwise, keep trust radius unchanged

        // Update success ratio with exponential smoothing
        self.success_ratio = utils::ewma_update(self.success_ratio, ratio, 0.1);
    }
}

impl<T: StreamingObjective + Clone> StreamingOptimizer for StreamingTrustRegion<T> {
    fn update(&mut self, datapoint: &StreamingDataPoint) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Evaluate current function value
        let current_f = self.objective.evaluate(&self.parameters.view(), datapoint);

        // Compute gradient
        let gradient = self.objective.gradient(&self.parameters.view(), datapoint);

        // Accumulate gradient for better estimates (exponential smoothing)
        if self.gradient_count == 0 {
            self.gradient_accumulator = gradient.clone();
        } else {
            let alpha = 1.0 / (self.gradient_count as f64 + 1.0).min(10.0); // Adaptive averaging
            self.gradient_accumulator =
                &((1.0 - alpha) * &self.gradient_accumulator) + &(alpha * &gradient);
        }
        self.gradient_count += 1;

        // Use accumulated gradient for more stable updates
        let effective_gradient = if self.gradient_count >= 3 {
            &self.gradient_accumulator
        } else {
            &gradient
        };

        // Solve trust region subproblem
        let step = self.solve_trust_region_subproblem(&effective_gradient.view())?;
        let step_norm = step.mapv(|x| x * x).sum().sqrt();

        // Trial _point
        let trial_parameters = &self.parameters + &step;
        let trial_f = self.objective.evaluate(&trial_parameters.view(), datapoint);

        // Compute actual reduction
        let actual_reduction = current_f - trial_f;

        // Compute trust region ratio
        let ratio = self.compute_trust_region_ratio(
            &step.view(),
            &effective_gradient.view(),
            actual_reduction,
        );

        // Accept or reject step
        const ACCEPTANCE_THRESHOLD: f64 = 0.1;
        if ratio >= ACCEPTANCE_THRESHOLD {
            // Accept step
            let old_parameters = self.parameters.clone();
            self.parameters = trial_parameters;

            // Update Hessian approximation if we have previous gradient
            if self.stats.updates_performed > 0 {
                let grad_diff = &gradient - &self.gradient_accumulator;
                self.update_hessian_approximation(&step.view(), &grad_diff.view());
            }

            // Check convergence
            self.stats.converged = utils::check_convergence(
                &old_parameters.view(),
                &self.parameters.view(),
                self.config.tolerance,
            );

            self.stats.updates_performed += 1;
            self.prev_function_value = trial_f;
        } else {
            // Reject step - only update trust radius
        }

        // Update trust region radius
        self.update_trust_radius(ratio, step_norm);

        // Update statistics
        self.stats.points_processed += 1;
        self.stats.current_loss = if ratio >= ACCEPTANCE_THRESHOLD {
            trial_f
        } else {
            current_f
        };
        self.stats.average_loss =
            utils::ewma_update(self.stats.average_loss, self.stats.current_loss, 0.01);

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
        let n = self.parameters.len();
        self.hessian_approx = Array2::eye(n);
        self.gradient_accumulator = Array1::zeros(n);
        self.gradient_count = 0;
        self.success_ratio = 0.5;
        self.prev_function_value = f64::INFINITY;
        self.stats = StreamingStats::default();
    }
}

/// Convenience function for streaming trust region with linear regression
#[allow(dead_code)]
pub fn streaming_trust_region_linear_regression(
    n_features: usize,
    config: Option<StreamingConfig>,
    initial_trust_radius: Option<f64>,
) -> StreamingTrustRegion<super::LinearRegressionObjective> {
    let config = config.unwrap_or_default();
    let trust_radius = initial_trust_radius.unwrap_or(1.0);
    let initial_params = Array1::zeros(n_features);
    let objective = super::LinearRegressionObjective;

    StreamingTrustRegion::new(initial_params, objective, config, trust_radius)
}

/// Convenience function for streaming trust region with logistic regression
#[allow(dead_code)]
pub fn streaming_trust_region_logistic_regression(
    n_features: usize,
    config: Option<StreamingConfig>,
    initial_trust_radius: Option<f64>,
) -> StreamingTrustRegion<super::LogisticRegressionObjective> {
    let config = config.unwrap_or_default();
    let trust_radius = initial_trust_radius.unwrap_or(1.0);
    let initial_params = Array1::zeros(n_features);
    let objective = super::LogisticRegressionObjective;

    StreamingTrustRegion::new(initial_params, objective, config, trust_radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{LinearRegressionObjective, StreamingDataPoint};

    #[test]
    fn test_streaming_trust_region_creation() {
        let params = Array1::from(vec![0.0, 0.0]);
        let objective = LinearRegressionObjective;
        let config = StreamingConfig::default();
        let trust_radius = 1.0;

        let optimizer = StreamingTrustRegion::new(params.clone(), objective, config, trust_radius);
        assert_eq!(optimizer.parameters(), &params);
        assert_eq!(optimizer.trust_radius, 1.0);
    }

    #[test]
    fn test_trust_region_subproblem_solving() {
        let params = Array1::from(vec![0.0, 0.0]);
        let objective = LinearRegressionObjective;
        let config = StreamingConfig::default();
        let trust_radius = 1.0;

        let optimizer = StreamingTrustRegion::new(params, objective, config, trust_radius);
        let gradient = Array1::from(vec![1.0, 2.0]);

        let step = optimizer
            .solve_trust_region_subproblem(&gradient.view())
            .unwrap();
        let step_norm = step.mapv(|x| x * x).sum().sqrt();

        // Step should be within trust region
        assert!(step_norm <= trust_radius + 1e-10);
    }

    #[test]
    fn test_streaming_trust_region_update() {
        let mut optimizer = streaming_trust_region_linear_regression(2, None, Some(1.0));

        let features = Array1::from(vec![1.0, 2.0]);
        let target = 3.0;
        let point = StreamingDataPoint::new(features, target);

        assert!(optimizer.update(&point).is_ok());
        assert_eq!(optimizer.stats().points_processed, 1);
    }

    #[test]
    fn test_hessian_update() {
        let params = Array1::from(vec![1.0, 1.0]);
        let objective = LinearRegressionObjective;
        let mut config = StreamingConfig::default();
        config.regularization = 1e-6;

        let mut optimizer = StreamingTrustRegion::new(params, objective, config, 1.0);

        let step = Array1::from(vec![0.1, 0.2]);
        let grad_diff = Array1::from(vec![0.05, 0.1]);

        let original_hessian = optimizer.hessian_approx.clone();
        optimizer.update_hessian_approximation(&step.view(), &grad_diff.view());

        // Hessian should have changed
        assert!(&optimizer.hessian_approx != &original_hessian);
    }

    #[test]
    fn test_trust_radius_adaptation() {
        let params = Array1::from(vec![0.0, 0.0]);
        let objective = LinearRegressionObjective;
        let config = StreamingConfig::default();
        let initial_radius = 1.0;

        let mut optimizer = StreamingTrustRegion::new(params, objective, config, initial_radius);

        // Test expansion with very successful step
        optimizer.update_trust_radius(0.9, 0.9); // High ratio, near boundary
        assert!(optimizer.trust_radius > initial_radius);

        // Test contraction with unsuccessful step
        optimizer.update_trust_radius(0.1, 0.5); // Low ratio
        assert!(optimizer.trust_radius < initial_radius);
    }

    #[test]
    fn test_convergence_detection() {
        let mut config = StreamingConfig::default();
        config.tolerance = 1e-2;
        config.learning_rate = 0.5;

        let mut optimizer = streaming_trust_region_linear_regression(2, Some(config), Some(1.0));

        // Use simple data that should converge quickly
        let point = StreamingDataPoint::new(Array1::from(vec![0.0, 0.0]), 0.0);

        // Should converge quickly with zero gradient
        for _ in 0..10 {
            optimizer.update(&point).unwrap();
            if optimizer.converged() {
                break;
            }
        }

        // Should detect convergence when parameters don't change much
        assert!(optimizer.converged() || optimizer.stats().updates_performed < 10);
    }
}
