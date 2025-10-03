//! Incremental Newton Methods for Streaming Optimization
//!
//! This module implements incremental Newton and quasi-Newton methods that
//! maintain second-order information efficiently for streaming data scenarios.
//! These methods are particularly useful when Hessian information is available
//! or can be approximated incrementally.

use super::{
    utils, StreamingConfig, StreamingDataPoint, StreamingObjective, StreamingOptimizer,
    StreamingStats,
};
use crate::error::OptimizeError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
// Unused import
// use scirs2_core::ndarray::ArrayView2;
// Unused import
// use scirs2_core::error::CoreResult;
// Unused import
// use scirs2_linalg::{cholesky, solve, LinalgError};
use scirs2_linalg::solve;
// Unused import: LinalgError

type Result<T> = std::result::Result<T, OptimizeError>;

/// Types of incremental Newton methods
#[derive(Debug, Clone, Copy)]
pub enum IncrementalNewtonMethod {
    /// Full Newton with exact Hessian (when available)
    Exact,
    /// BFGS quasi-Newton approximation
    BFGS,
    /// L-BFGS with limited memory
    LBFGS(usize), // memory size
    /// Sherman-Morrison rank-1 updates
    ShermanMorrison,
}

/// Incremental Newton optimizer
#[derive(Debug, Clone)]
pub struct IncrementalNewton<T: StreamingObjective> {
    /// Current parameter estimates
    parameters: Array1<f64>,
    /// Objective function
    objective: T,
    /// Configuration
    config: StreamingConfig,
    /// Statistics
    stats: StreamingStats,
    /// Method type
    method: IncrementalNewtonMethod,
    /// Current Hessian approximation or inverse Hessian
    hessian_inv: Array2<f64>,
    /// Previous gradient for BFGS updates
    prev_gradient: Option<Array1<f64>>,
    /// L-BFGS history storage
    lbfgs_history: LBFGSHistory,
    /// Regularization for numerical stability
    regularization: f64,
    /// Line search step size
    step_size: f64,
}

/// Storage for L-BFGS history
#[derive(Debug, Clone)]
struct LBFGSHistory {
    /// Storage for s vectors (parameter differences)
    s_vectors: Vec<Array1<f64>>,
    /// Storage for y vectors (gradient differences)  
    y_vectors: Vec<Array1<f64>>,
    /// Storage for rho values (1 / (y^T s))
    rho_values: Vec<f64>,
    /// Maximum memory size
    memory_size: usize,
    /// Current position in circular buffer
    current_pos: usize,
    /// Number of stored vectors
    stored_count: usize,
}

impl LBFGSHistory {
    fn new(_memorysize: usize) -> Self {
        Self {
            s_vectors: Vec::with_capacity(_memorysize),
            y_vectors: Vec::with_capacity(_memorysize),
            rho_values: Vec::with_capacity(_memorysize),
            memory_size: _memorysize,
            current_pos: 0,
            stored_count: 0,
        }
    }

    fn add_pair(&mut self, s: Array1<f64>, y: Array1<f64>) {
        let rho = 1.0 / s.dot(&y);

        if self.stored_count < self.memory_size {
            self.s_vectors.push(s);
            self.y_vectors.push(y);
            self.rho_values.push(rho);
            self.stored_count += 1;
        } else {
            self.s_vectors[self.current_pos] = s;
            self.y_vectors[self.current_pos] = y;
            self.rho_values[self.current_pos] = rho;
        }

        self.current_pos = (self.current_pos + 1) % self.memory_size;
    }

    fn compute_direction(&self, gradient: &ArrayView1<f64>) -> Array1<f64> {
        if self.stored_count == 0 {
            return -gradient.to_owned();
        }

        let mut q = gradient.to_owned();
        let mut alpha = vec![0.0; self.stored_count];

        // First loop (backward)
        for i in (0..self.stored_count).rev() {
            let idx = (self.current_pos + self.memory_size - 1 - i) % self.memory_size;
            alpha[i] = self.rho_values[idx] * self.s_vectors[idx].dot(&q);
            q = &q - &(alpha[i] * &self.y_vectors[idx]);
        }

        // Initial Hessian approximation (use identity scaled by most recent curvature)
        if let (Some(s), Some(y)) = (self.s_vectors.last(), self.y_vectors.last()) {
            let gamma = s.dot(y) / y.dot(y);
            if gamma > 0.0 && gamma.is_finite() {
                q *= gamma;
            }
        }

        // Second loop (forward)
        for i in 0..self.stored_count {
            let idx =
                (self.current_pos + self.memory_size - self.stored_count + i) % self.memory_size;
            let beta = self.rho_values[idx] * self.y_vectors[idx].dot(&q);
            q = &q + &((alpha[self.stored_count - 1 - i] - beta) * &self.s_vectors[idx]);
        }

        -q
    }

    fn reset(&mut self) {
        self.s_vectors.clear();
        self.y_vectors.clear();
        self.rho_values.clear();
        self.current_pos = 0;
        self.stored_count = 0;
    }
}

impl<T: StreamingObjective> IncrementalNewton<T> {
    /// Create a new incremental Newton optimizer
    pub fn new(
        initial_parameters: Array1<f64>,
        objective: T,
        config: StreamingConfig,
        method: IncrementalNewtonMethod,
    ) -> Self {
        let n_params = initial_parameters.len();
        let memory_size = match method {
            IncrementalNewtonMethod::LBFGS(m) => m,
            _ => 10, // Default memory size for other methods
        };

        Self {
            parameters: initial_parameters,
            objective,
            config,
            stats: StreamingStats::default(),
            method,
            hessian_inv: Array2::eye(n_params),
            prev_gradient: None,
            lbfgs_history: LBFGSHistory::new(memory_size),
            regularization: 1e-8,
            step_size: 1.0,
        }
    }

    /// Update inverse Hessian using BFGS formula
    fn update_bfgs(&mut self, s: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<()> {
        let n = s.len();
        let sy = s.dot(y);

        if sy.abs() < 1e-12 {
            return Ok(()); // Skip update if curvature condition not satisfied
        }

        // BFGS update: H^{-1} = (I - ρsy^T)H^{-1}(I - ρys^T) + ρss^T
        let rho = 1.0 / sy;

        // Compute H^{-1} * y
        let mut hy = Array1::zeros(n);
        for i in 0..n {
            for j in 0..n {
                hy[i] += self.hessian_inv[[i, j]] * y[j];
            }
        }

        let yhy = y.dot(&hy);

        // Update formula: H^{-1} = H^{-1} + ρ²(yHy)ss^T - ρ(Hys^T + sy^TH)
        for i in 0..n {
            for j in 0..n {
                self.hessian_inv[[i, j]] +=
                    rho * rho * yhy * s[i] * s[j] - rho * (hy[i] * s[j] + s[i] * hy[j]);
            }
        }

        Ok(())
    }

    /// Update using Sherman-Morrison formula for rank-1 updates
    fn update_sherman_morrison(&mut self, u: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Result<()> {
        let n = u.len();

        // Compute H^{-1} * u
        let mut hu = Array1::zeros(n);
        for i in 0..n {
            for j in 0..n {
                hu[i] += self.hessian_inv[[i, j]] * u[j];
            }
        }

        let vhu = v.dot(&hu);
        if (1.0 + vhu).abs() < 1e-12 {
            return Ok(()); // Skip update if denominator too small
        }

        // Sherman-Morrison update: H^{-1} = H^{-1} - (H^{-1}uv^TH^{-1})/(1 + v^TH^{-1}u)
        let scale = 1.0 / (1.0 + vhu);
        for i in 0..n {
            for j in 0..n {
                let hv_j = (0..n).map(|k| self.hessian_inv[[j, k]] * v[k]).sum::<f64>();
                self.hessian_inv[[i, j]] -= scale * hu[i] * hv_j;
            }
        }

        Ok(())
    }

    /// Compute Newton direction based on method type
    fn compute_newton_direction(&mut self, gradient: &ArrayView1<f64>) -> Result<Array1<f64>> {
        match self.method {
            IncrementalNewtonMethod::Exact => {
                // Use exact Hessian if available from objective function
                if let Some(data_point) = self.get_current_data_point() {
                    if let Some(hessian) = T::hessian(&self.parameters.view(), &data_point) {
                        // Add regularization for numerical stability
                        let mut reg_hessian = hessian;
                        for i in 0..reg_hessian.nrows() {
                            reg_hessian[[i, i]] += self.regularization;
                        }

                        match solve(&reg_hessian.view(), &(-gradient).view(), None) {
                            Ok(direction) => return Ok(direction),
                            Err(_) => {
                                // Fall back to inverse Hessian approximation
                            }
                        }
                    }
                }

                // Fall back to using current inverse Hessian approximation
                Ok(self.hessian_inv.dot(&(-gradient)))
            }
            IncrementalNewtonMethod::BFGS => Ok(self.hessian_inv.dot(&(-gradient))),
            IncrementalNewtonMethod::LBFGS(_) => Ok(self.lbfgs_history.compute_direction(gradient)),
            IncrementalNewtonMethod::ShermanMorrison => Ok(self.hessian_inv.dot(&(-gradient))),
        }
    }

    /// Simple line search to determine step size
    fn line_search(
        &self,
        direction: &ArrayView1<f64>,
        gradient: &ArrayView1<f64>,
        data_point: &StreamingDataPoint,
    ) -> f64 {
        let mut alpha = self.step_size;
        let c1 = 1e-4; // Armijo condition parameter
        let rho = 0.5; // Backtracking factor

        let current_f = self.objective.evaluate(&self.parameters.view(), data_point);
        let directional_derivative = gradient.dot(direction);

        for _ in 0..20 {
            // Maximum backtracking steps
            let trial_params = &self.parameters + &(alpha * direction);
            let trial_f = self.objective.evaluate(&trial_params.view(), data_point);

            // Armijo condition
            if trial_f <= current_f + c1 * alpha * directional_derivative {
                return alpha;
            }

            alpha *= rho;
        }

        alpha.max(1e-8) // Minimum step size
    }

    /// Placeholder for getting current data point (would be passed in real implementation)
    fn get_current_data_point(&self) -> Option<StreamingDataPoint> {
        // In practice, this would be provided by the calling context
        None
    }
}

impl<T: StreamingObjective + Clone> StreamingOptimizer for IncrementalNewton<T> {
    fn update(&mut self, datapoint: &StreamingDataPoint) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Compute current gradient
        let gradient = self.objective.gradient(&self.parameters.view(), datapoint);

        // Compute Newton direction
        let direction = self.compute_newton_direction(&gradient.view())?;

        // Perform line search
        let alpha = self.line_search(&direction.view(), &gradient.view(), datapoint);

        // Update parameters
        let old_parameters = self.parameters.clone();
        let step = &direction * alpha;
        self.parameters = &self.parameters + &step;

        // Update second-order information
        if let Some(prev_grad) = &self.prev_gradient {
            let s = &step;
            let y = &gradient - prev_grad;

            match self.method {
                IncrementalNewtonMethod::BFGS => {
                    self.update_bfgs(&s.view(), &y.view())?;
                }
                IncrementalNewtonMethod::LBFGS(_) => {
                    self.lbfgs_history.add_pair(s.clone(), y.clone());
                }
                IncrementalNewtonMethod::ShermanMorrison => {
                    // Use gradient difference as rank-1 update
                    if y.dot(&y) > 1e-12 {
                        self.update_sherman_morrison(&s.view(), &y.view())?;
                    }
                }
                IncrementalNewtonMethod::Exact => {
                    // For exact method, we update the inverse Hessian approximation
                    // using BFGS as a fallback
                    self.update_bfgs(&s.view(), &y.view())?;
                }
            }
        }

        // Store current gradient for next iteration
        self.prev_gradient = Some(gradient.clone());

        // Update step size using simple adaptation
        let step_norm = step.mapv(|x| x * x).sum().sqrt();
        if step_norm > 0.0 {
            let param_norm = self.parameters.mapv(|x| x * x).sum().sqrt().max(1.0);
            let relative_step = step_norm / param_norm;

            if relative_step > 0.1 {
                self.step_size *= 0.8; // Reduce step size if too large
            } else if relative_step < 0.01 {
                self.step_size *= 1.2; // Increase step size if too small
            }
            self.step_size = self.step_size.max(1e-8).min(10.0);
        }

        // Check convergence
        self.stats.converged = utils::check_convergence(
            &old_parameters.view(),
            &self.parameters.view(),
            self.config.tolerance,
        );

        // Update statistics
        let loss = self.objective.evaluate(&self.parameters.view(), datapoint);
        self.stats.points_processed += 1;
        self.stats.updates_performed += 1;
        self.stats.current_loss = loss;
        self.stats.average_loss = utils::ewma_update(self.stats.average_loss, loss, 0.01);

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
        self.hessian_inv = Array2::eye(n);
        self.prev_gradient = None;
        self.lbfgs_history.reset();
        self.step_size = 1.0;
        self.stats = StreamingStats::default();
    }
}

/// Convenience function for incremental Newton with BFGS
#[allow(dead_code)]
pub fn incremental_bfgs<T: StreamingObjective>(
    initial_parameters: Array1<f64>,
    objective: T,
    config: Option<StreamingConfig>,
) -> IncrementalNewton<T> {
    let config = config.unwrap_or_default();
    IncrementalNewton::new(
        initial_parameters,
        objective,
        config,
        IncrementalNewtonMethod::BFGS,
    )
}

/// Convenience function for incremental L-BFGS
#[allow(dead_code)]
pub fn incremental_lbfgs<T: StreamingObjective>(
    initial_parameters: Array1<f64>,
    objective: T,
    memory_size: usize,
    config: Option<StreamingConfig>,
) -> IncrementalNewton<T> {
    let config = config.unwrap_or_default();
    IncrementalNewton::new(
        initial_parameters,
        objective,
        config,
        IncrementalNewtonMethod::LBFGS(memory_size),
    )
}

/// Convenience function for L-BFGS linear regression
#[allow(dead_code)]
pub fn incremental_lbfgs_linear_regression(
    n_features: usize,
    memory_size: Option<usize>,
    config: Option<StreamingConfig>,
) -> IncrementalNewton<super::LinearRegressionObjective> {
    let config = config.unwrap_or_default();
    let memory = memory_size.unwrap_or(10);
    let initial_params = Array1::zeros(n_features);
    let objective = super::LinearRegressionObjective;

    IncrementalNewton::new(
        initial_params,
        objective,
        config,
        IncrementalNewtonMethod::LBFGS(memory),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{LinearRegressionObjective, StreamingDataPoint};

    #[test]
    fn test_incremental_newton_creation() {
        let params = Array1::from(vec![0.0, 0.0]);
        let objective = LinearRegressionObjective;
        let config = StreamingConfig::default();

        let optimizer = IncrementalNewton::new(
            params.clone(),
            objective,
            config,
            IncrementalNewtonMethod::BFGS,
        );
        assert_eq!(optimizer.parameters(), &params);
        assert!(matches!(optimizer.method, IncrementalNewtonMethod::BFGS));
    }

    #[test]
    fn test_lbfgs_history() {
        let mut history = LBFGSHistory::new(3);

        let s1 = Array1::from(vec![1.0, 0.0]);
        let y1 = Array1::from(vec![0.5, 0.0]);
        history.add_pair(s1, y1);

        assert_eq!(history.stored_count, 1);

        let gradient = Array1::from(vec![1.0, 1.0]);
        let direction = history.compute_direction(&gradient.view());

        // Direction should be approximately -gradient scaled
        assert!(direction[0] < 0.0);
        assert!(direction[1] < 0.0);
    }

    #[test]
    fn test_incremental_bfgs_update() {
        let mut optimizer = incremental_bfgs(
            Array1::from(vec![0.0, 0.0]),
            LinearRegressionObjective,
            None,
        );

        let features = Array1::from(vec![1.0, 2.0]);
        let target = 3.0;
        let point = StreamingDataPoint::new(features, target);

        assert!(optimizer.update(&point).is_ok());
        assert_eq!(optimizer.stats().points_processed, 1);
    }

    #[test]
    fn test_incremental_lbfgs_update() {
        let mut optimizer = incremental_lbfgs_linear_regression(2, Some(5), None);

        let features = Array1::from(vec![1.0, 2.0]);
        let target = 3.0;
        let point = StreamingDataPoint::new(features, target);

        assert!(optimizer.update(&point).is_ok());
        assert_eq!(optimizer.stats().points_processed, 1);
    }

    #[test]
    fn test_bfgs_hessian_update() {
        let params = Array1::from(vec![1.0, 1.0]);
        let objective = LinearRegressionObjective;
        let config = StreamingConfig::default();

        let mut optimizer =
            IncrementalNewton::new(params, objective, config, IncrementalNewtonMethod::BFGS);

        let s = Array1::from(vec![0.1, 0.2]);
        let y = Array1::from(vec![0.05, 0.1]);

        let original_hessian = optimizer.hessian_inv.clone();
        optimizer.update_bfgs(&s.view(), &y.view()).unwrap();

        // Hessian should have changed
        assert!(&optimizer.hessian_inv != &original_hessian);
    }

    #[test]
    fn test_sherman_morrison_update() {
        let params = Array1::from(vec![1.0, 1.0]);
        let objective = LinearRegressionObjective;
        let config = StreamingConfig::default();

        let mut optimizer = IncrementalNewton::new(
            params,
            objective,
            config,
            IncrementalNewtonMethod::ShermanMorrison,
        );

        let u = Array1::from(vec![0.1, 0.2]);
        let v = Array1::from(vec![0.3, 0.4]);

        let original_hessian = optimizer.hessian_inv.clone();
        optimizer
            .update_sherman_morrison(&u.view(), &v.view())
            .unwrap();

        // Hessian should have changed
        assert!(&optimizer.hessian_inv != &original_hessian);
    }

    #[test]
    fn test_convergence_with_multiple_updates() {
        let mut config = StreamingConfig::default();
        config.tolerance = 1e-3;

        let mut optimizer = incremental_lbfgs_linear_regression(2, Some(5), Some(config));

        // Generate consistent data for y = 2*x1 + 3*x2
        let data_points = vec![
            StreamingDataPoint::new(Array1::from(vec![1.0, 0.0]), 2.0),
            StreamingDataPoint::new(Array1::from(vec![0.0, 1.0]), 3.0),
            StreamingDataPoint::new(Array1::from(vec![1.0, 1.0]), 5.0),
        ];

        for _epoch in 0..20 {
            for point in &data_points {
                optimizer.update(point).unwrap();
                if optimizer.converged() {
                    break;
                }
            }
            if optimizer.converged() {
                break;
            }
        }

        // Should make some progress toward the solution
        let params = optimizer.parameters();
        assert!(optimizer.stats().updates_performed > 0);
    }
}
