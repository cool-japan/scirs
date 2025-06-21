//! Acceleration methods for iterative algorithms
//!
//! This module provides various acceleration techniques to improve the convergence
//! of fixed-point iterations, nonlinear solvers, and other iterative methods.
//! These methods are particularly useful for implicit ODE/DAE solvers and
//! iterative linear/nonlinear equation solvers.
//!
//! # Anderson Acceleration
//!
//! Anderson acceleration (also known as Anderson mixing) is a technique for
//! accelerating fixed-point iterations x_{k+1} = G(x_k) by using information
//! from previous iterates to extrapolate to better solutions.
//!
//! # Examples
//!
//! ```
//! use scirs2_integrate::acceleration::{AndersonAccelerator, AcceleratorOptions};
//! use ndarray::Array1;
//!
//! // Create accelerator for 3D problem with memory depth 5
//! let mut accelerator = AndersonAccelerator::new(3, AcceleratorOptions::default());
//!
//! // In your iteration loop:
//! let x_current = Array1::from_vec(vec![1.0, 2.0, 3.0]);
//! let g_x = Array1::from_vec(vec![1.1, 1.9, 3.1]); // G(x_current)
//!
//! // Get accelerated update
//! if let Some(x_accelerated) = accelerator.accelerate(x_current.view(), g_x.view()) {
//!     // Use x_accelerated for next iteration
//! }
//! ```

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use ndarray::{Array1, Array2, ArrayView1};
use std::collections::VecDeque;

/// Options for acceleration methods
#[derive(Debug, Clone)]
pub struct AcceleratorOptions<F: IntegrateFloat> {
    /// Maximum number of previous iterates to store (memory depth)
    pub memory_depth: usize,
    /// Regularization parameter for least squares problems
    pub regularization: F,
    /// Minimum number of iterates before starting acceleration
    pub min_iterates: usize,
    /// Whether to use QR decomposition for better numerical stability
    pub use_qr: bool,
    /// Damping factor for Anderson acceleration (0 < damping ≤ 1)
    pub damping: F,
    /// Whether to restart acceleration periodically
    pub restart_period: Option<usize>,
}

impl<F: IntegrateFloat> Default for AcceleratorOptions<F> {
    fn default() -> Self {
        Self {
            memory_depth: 5,
            regularization: F::from(1e-12).unwrap(),
            min_iterates: 2,
            use_qr: true,
            damping: F::one(),
            restart_period: Some(20),
        }
    }
}

/// Anderson accelerator for fixed-point iterations
///
/// Accelerates iterations of the form x_{k+1} = G(x_k) by maintaining
/// a history of iterates and residuals, then solving a least-squares
/// problem to find optimal linear combination coefficients.
#[derive(Debug)]
pub struct AndersonAccelerator<F: IntegrateFloat> {
    /// Problem dimension
    dimension: usize,
    /// Configuration options
    options: AcceleratorOptions<F>,
    /// History of iterates (x_k)
    x_history: VecDeque<Array1<F>>,
    /// History of function values (G(x_k))
    g_history: VecDeque<Array1<F>>,
    /// History of residuals (G(x_k) - x_k)
    residual_history: VecDeque<Array1<F>>,
    /// Current iteration count
    iteration_count: usize,
    /// Whether acceleration is active
    is_active: bool,
}

impl<F: IntegrateFloat> AndersonAccelerator<F> {
    /// Create a new Anderson accelerator
    pub fn new(dimension: usize, options: AcceleratorOptions<F>) -> Self {
        Self {
            dimension,
            options,
            x_history: VecDeque::new(),
            g_history: VecDeque::new(),
            residual_history: VecDeque::new(),
            iteration_count: 0,
            is_active: false,
        }
    }

    /// Create accelerator with default options
    pub fn with_memory_depth(dimension: usize, memory_depth: usize) -> Self {
        let options = AcceleratorOptions {
            memory_depth,
            ..Default::default()
        };
        Self::new(dimension, options)
    }

    /// Add a new iterate and return accelerated update if possible
    pub fn accelerate(
        &mut self,
        x_current: ArrayView1<F>,
        g_x_current: ArrayView1<F>,
    ) -> Option<Array1<F>> {
        if x_current.len() != self.dimension || g_x_current.len() != self.dimension {
            return None;
        }

        // Compute residual: r_k = G(x_k) - x_k
        let residual = &g_x_current.to_owned() - &x_current.to_owned();

        // Store current iterate
        self.x_history.push_back(x_current.to_owned());
        self.g_history.push_back(g_x_current.to_owned());
        self.residual_history.push_back(residual);

        // Maintain memory depth
        while self.x_history.len() > self.options.memory_depth {
            self.x_history.pop_front();
            self.g_history.pop_front();
            self.residual_history.pop_front();
        }

        self.iteration_count += 1;

        // Check if we should restart
        if let Some(restart_period) = self.options.restart_period {
            if self.iteration_count % restart_period == 0 {
                self.restart();
                return Some(g_x_current.to_owned());
            }
        }

        // Need at least min_iterates to start acceleration
        if self.residual_history.len() < self.options.min_iterates {
            return Some(g_x_current.to_owned());
        }

        // Attempt Anderson acceleration
        self.is_active = true;
        match self.compute_anderson_update() {
            Ok(x_accelerated) => Some(x_accelerated),
            Err(_) => {
                // Fallback to unaccelerated update
                self.restart();
                Some(g_x_current.to_owned())
            }
        }
    }

    /// Compute Anderson accelerated update
    fn compute_anderson_update(&self) -> IntegrateResult<Array1<F>> {
        let m = self.residual_history.len() - 1; // Number of residual differences

        if m == 0 {
            // Not enough history for acceleration
            return Ok(self.g_history.back().unwrap().clone());
        }

        // Build residual difference matrix: ΔR = [r_1 - r_0, r_2 - r_1, ..., r_m - r_{m-1}]
        let mut delta_r = Array2::zeros((self.dimension, m));

        for j in 0..m {
            let r_diff = &self.residual_history[j + 1] - &self.residual_history[j];
            for i in 0..self.dimension {
                delta_r[[i, j]] = r_diff[i];
            }
        }

        // Solve least squares problem: min_α ||ΔR α + r_m||²
        let r_m = self.residual_history.back().unwrap();
        let alpha = self.solve_least_squares(&delta_r, r_m.view())?;

        // Compute accelerated update
        let mut x_accelerated = Array1::zeros(self.dimension);
        let mut g_accelerated = Array1::zeros(self.dimension);

        // x_acc = (1 - Σα_j) x_m + Σα_j x_j
        // g_acc = (1 - Σα_j) G(x_m) + Σα_j G(x_j)
        let alpha_sum: F = alpha.sum();
        let weight_m = F::one() - alpha_sum;

        // Add contribution from most recent iterate
        let x_m = self.x_history.back().unwrap();
        let g_m = self.g_history.back().unwrap();

        x_accelerated = &x_accelerated + &(x_m * weight_m);
        g_accelerated = &g_accelerated + &(g_m * weight_m);

        // Add contributions from historical iterates
        for (j, alpha_j) in alpha.iter().enumerate() {
            x_accelerated = &x_accelerated + &(&self.x_history[j] * *alpha_j);
            g_accelerated = &g_accelerated + &(&self.g_history[j] * *alpha_j);
        }

        // Apply damping: x_new = (1-β) x_acc + β g_acc
        let beta = self.options.damping;
        let x_new = &x_accelerated * (F::one() - beta) + &g_accelerated * beta;

        Ok(x_new)
    }

    /// Solve least squares problem using QR decomposition or normal equations
    fn solve_least_squares(&self, a: &Array2<F>, b: ArrayView1<F>) -> IntegrateResult<Array1<F>> {
        let (n, m) = a.dim();

        if self.options.use_qr && n >= m {
            self.solve_qr(a, b)
        } else {
            self.solve_normal_equations(a, b)
        }
    }

    /// Solve using QR decomposition (more stable for tall matrices)
    fn solve_qr(&self, a: &Array2<F>, b: ArrayView1<F>) -> IntegrateResult<Array1<F>> {
        // For now, fallback to normal equations
        // Full QR implementation would require more advanced linear algebra
        self.solve_normal_equations(a, b)
    }

    /// Solve using normal equations: (A^T A + λI) x = A^T b
    fn solve_normal_equations(
        &self,
        a: &Array2<F>,
        b: ArrayView1<F>,
    ) -> IntegrateResult<Array1<F>> {
        let (n, m) = a.dim();

        // Compute A^T A
        let mut ata = Array2::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let mut sum = F::zero();
                for k in 0..n {
                    sum += a[[k, i]] * a[[k, j]];
                }
                ata[[i, j]] = sum;
            }
        }

        // Add regularization: A^T A + λI
        for i in 0..m {
            ata[[i, i]] += self.options.regularization;
        }

        // Compute A^T b
        let mut atb = Array1::zeros(m);
        for i in 0..m {
            let mut sum = F::zero();
            for k in 0..n {
                sum += a[[k, i]] * b[k];
            }
            atb[i] = sum;
        }

        // Solve linear system using Gaussian elimination
        self.solve_linear_system(ata, atb)
    }

    /// Solve linear system using Gaussian elimination with partial pivoting
    fn solve_linear_system(
        &self,
        mut a: Array2<F>,
        mut b: Array1<F>,
    ) -> IntegrateResult<Array1<F>> {
        let n = a.nrows();

        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_val = a[[k, k]].abs();
            let mut max_row = k;

            for i in (k + 1)..n {
                let abs_val = a[[i, k]].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    max_row = i;
                }
            }

            // Check for singular matrix
            if max_val < self.options.regularization {
                return Err(IntegrateError::ComputationError(
                    "Singular matrix in Anderson acceleration".to_string(),
                ));
            }

            // Swap rows if needed
            if max_row != k {
                for j in 0..n {
                    let temp = a[[k, j]];
                    a[[k, j]] = a[[max_row, j]];
                    a[[max_row, j]] = temp;
                }
                let temp = b[k];
                b[k] = b[max_row];
                b[max_row] = temp;
            }

            // Elimination
            for i in (k + 1)..n {
                let factor = a[[i, k]] / a[[k, k]];
                for j in (k + 1)..n {
                    a[[i, j]] = a[[i, j]] - factor * a[[k, j]];
                }
                b[i] = b[i] - factor * b[k];
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = F::zero();
            for j in (i + 1)..n {
                sum += a[[i, j]] * x[j];
            }
            x[i] = (b[i] - sum) / a[[i, i]];
        }

        Ok(x)
    }

    /// Restart the accelerator (clear history)
    pub fn restart(&mut self) {
        self.x_history.clear();
        self.g_history.clear();
        self.residual_history.clear();
        self.is_active = false;
    }

    /// Check if acceleration is currently active
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Get current memory usage (number of stored iterates)
    pub fn memory_usage(&self) -> usize {
        self.x_history.len()
    }

    /// Get iteration count
    pub fn iteration_count(&self) -> usize {
        self.iteration_count
    }
}

/// Simplified Aitken acceleration for scalar sequences
pub struct AitkenAccelerator<F: IntegrateFloat> {
    history: VecDeque<F>,
}

impl<F: IntegrateFloat> AitkenAccelerator<F> {
    /// Create new Aitken accelerator
    pub fn new() -> Self {
        Self {
            history: VecDeque::new(),
        }
    }

    /// Add new value and get accelerated estimate if possible
    pub fn accelerate(&mut self, value: F) -> Option<F> {
        self.history.push_back(value);

        // Keep only last 3 values
        while self.history.len() > 3 {
            self.history.pop_front();
        }

        if self.history.len() == 3 {
            let x0 = self.history[0];
            let x1 = self.history[1];
            let x2 = self.history[2];

            // Aitken formula: x_acc = x2 - (x2 - x1)² / (x2 - 2x1 + x0)
            let numerator = (x2 - x1) * (x2 - x1);
            let denominator = x2 - F::from(2.0).unwrap() * x1 + x0;

            if denominator.abs() > F::from(1e-12).unwrap() {
                Some(x2 - numerator / denominator)
            } else {
                Some(x2)
            }
        } else {
            Some(value)
        }
    }

    /// Restart the accelerator
    pub fn restart(&mut self) {
        self.history.clear();
    }
}

impl<F: IntegrateFloat> Default for AitkenAccelerator<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anderson_accelerator() {
        // Test simple fixed-point iteration: x_{k+1} = 0.5 * x_k + 1
        // Exact solution: x* = 2
        let mut accelerator = AndersonAccelerator::new(1, AcceleratorOptions::default());

        let mut x = Array1::from_vec(vec![0.0]);

        for _iter in 0..10 {
            let g_x = Array1::from_vec(vec![0.5 * x[0] + 1.0]);

            if let Some(x_new) = accelerator.accelerate(x.view(), g_x.view()) {
                x = x_new;
            }
        }

        // Should converge faster than unaccelerated iteration
        assert!((x[0] - 2.0_f64).abs() < 0.1);
    }

    #[test]
    fn test_aitken_accelerator() {
        let mut accelerator = AitkenAccelerator::new();

        // Test sequence converging to 1: x_n = 1 - 1/n
        let values = vec![0.0, 0.5, 0.666667, 0.75, 0.8];
        let mut result = 0.0;

        for value in values {
            if let Some(accelerated) = accelerator.accelerate(value) {
                result = accelerated;
            }
        }

        // Accelerated result should be closer to 1 than the last term
        assert!(result > 0.8);
    }

    #[test]
    fn test_anderson_with_regularization() {
        let options = AcceleratorOptions {
            regularization: 1e-8,
            memory_depth: 3,
            ..Default::default()
        };

        let mut accelerator = AndersonAccelerator::new(2, options);

        // Test 2D fixed-point iteration
        let mut x: Array1<f64> = Array1::from_vec(vec![0.0, 0.0]);

        for _iter in 0..5 {
            let g_x = Array1::from_vec(vec![
                0.3 * x[0] + 0.1 * x[1] + 1.0,
                0.1 * x[0] + 0.4 * x[1] + 0.5,
            ]);

            if let Some(x_new) = accelerator.accelerate(x.view(), g_x.view()) {
                x = x_new;
            }
        }

        // Check that solution is reasonable
        assert!(x[0].is_finite() && x[1].is_finite());
    }
}
