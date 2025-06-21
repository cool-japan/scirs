//! Second-order optimization methods
//!
//! This module provides implementations of second-order optimization methods
//! that use curvature information (Hessian matrix) to improve convergence.

use crate::error::{OptimError, Result};
use ndarray::{Array, Array1, Array2, Dimension, ScalarOperand};
use num_traits::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

/// Trait for second-order optimization methods
pub trait SecondOrderOptimizer<A: Float + ScalarOperand + Debug, D: Dimension> {
    /// Update parameters using second-order information
    fn step_second_order(
        &mut self,
        params: &Array<A, D>,
        gradients: &Array<A, D>,
        hessian_info: &HessianInfo<A, D>,
    ) -> Result<Array<A, D>>;

    /// Reset optimizer state
    fn reset(&mut self);
}

/// Hessian information for second-order methods
#[derive(Debug, Clone)]
pub enum HessianInfo<A: Float, D: Dimension> {
    /// Full Hessian matrix (expensive, rarely used in practice)
    Full(Array2<A>),
    /// Diagonal approximation of Hessian
    Diagonal(Array<A, D>),
    /// L-BFGS style quasi-Newton approximation
    QuasiNewton {
        /// Parameter differences history
        s_history: VecDeque<Array<A, D>>,
        /// Gradient differences history
        y_history: VecDeque<Array<A, D>>,
    },
    /// Gauss-Newton approximation for least squares problems
    GaussNewton(Array2<A>),
}

/// Approximated Hessian computation methods
pub mod hessian_approximation {
    use super::*;

    /// Compute diagonal Hessian approximation using finite differences (1D only)
    pub fn diagonal_finite_difference<A, F>(
        params: &Array1<A>,
        gradient_fn: F,
        epsilon: A,
    ) -> Result<Array1<A>>
    where
        A: Float + ScalarOperand + Debug + Copy,
        F: Fn(&Array1<A>) -> Result<Array1<A>>,
    {
        let mut hessian_diag = Array1::zeros(params.len());
        let _original_grad = gradient_fn(params)?;

        for i in 0..params.len() {
            let mut param_plus = params.clone();
            let mut param_minus = params.clone();

            // Forward difference: f(x + h) - f(x)
            param_plus[i] = params[i] + epsilon;
            let grad_plus = gradient_fn(&param_plus)?;

            // Backward difference: f(x) - f(x - h)
            param_minus[i] = params[i] - epsilon;
            let grad_minus = gradient_fn(&param_minus)?;

            // Hessian diagonal: derivative of gradient using central difference
            let second_deriv = (grad_plus[i] - grad_minus[i]) / (A::from(2.0).unwrap() * epsilon);
            hessian_diag[i] = second_deriv;
        }

        Ok(hessian_diag)
    }

    /// Update L-BFGS Hessian approximation
    pub fn update_lbfgs_approximation<A, D>(
        s_history: &mut VecDeque<Array<A, D>>,
        y_history: &mut VecDeque<Array<A, D>>,
        param_diff: Array<A, D>,
        grad_diff: Array<A, D>,
        max_history: usize,
    ) where
        A: Float + ScalarOperand + Debug,
        D: Dimension,
    {
        // Add new differences to history
        s_history.push_back(param_diff);
        y_history.push_back(grad_diff);

        // Maintain maximum history size
        if s_history.len() > max_history {
            s_history.pop_front();
            y_history.pop_front();
        }
    }

    /// Apply L-BFGS two-loop recursion to approximate H^(-1) * grad
    pub fn lbfgs_two_loop_recursion<A, D>(
        gradient: &Array<A, D>,
        s_history: &VecDeque<Array<A, D>>,
        y_history: &VecDeque<Array<A, D>>,
        initial_hessian_scale: A,
    ) -> Result<Array<A, D>>
    where
        A: Float + ScalarOperand + Debug,
        D: Dimension,
    {
        if s_history.len() != y_history.len() {
            return Err(OptimError::InvalidConfig(
                "History sizes don't match in L-BFGS".to_string(),
            ));
        }

        let m = s_history.len();
        if m == 0 {
            // No history, return scaled gradient
            return Ok(gradient * initial_hessian_scale);
        }

        let mut q = gradient.clone();
        let mut alphas = Vec::with_capacity(m);

        // First loop: compute alphas and update q
        for i in (0..m).rev() {
            let s_i = &s_history[i];
            let y_i = &y_history[i];

            // Compute rho_i = 1 / (y_i^T * s_i)
            let y_dot_s = y_i
                .iter()
                .zip(s_i.iter())
                .map(|(&y, &s)| y * s)
                .fold(A::zero(), |acc, x| acc + x);

            if y_dot_s.abs() < A::from(1e-12).unwrap() {
                alphas.push(A::zero());
                continue;
            }

            let rho_i = A::one() / y_dot_s;

            // Compute alpha_i = rho_i * s_i^T * q
            let s_dot_q = s_i
                .iter()
                .zip(q.iter())
                .map(|(&s, &q_val)| s * q_val)
                .fold(A::zero(), |acc, x| acc + x);
            let alpha_i = rho_i * s_dot_q;
            alphas.push(alpha_i);

            // Update q = q - alpha_i * y_i
            for (q_val, &y_val) in q.iter_mut().zip(y_i.iter()) {
                *q_val = *q_val - alpha_i * y_val;
            }
        }

        // Scale by initial Hessian approximation
        q.mapv_inplace(|x| x * initial_hessian_scale);

        // Second loop: compute final result
        alphas.reverse(); // Reverse to match forward iteration
        for i in 0..m {
            let s_i = &s_history[i];
            let y_i = &y_history[i];

            // Compute rho_i = 1 / (y_i^T * s_i)
            let y_dot_s = y_i
                .iter()
                .zip(s_i.iter())
                .map(|(&y, &s)| y * s)
                .fold(A::zero(), |acc, x| acc + x);

            if y_dot_s.abs() < A::from(1e-12).unwrap() {
                continue;
            }

            let rho_i = A::one() / y_dot_s;

            // Compute beta = rho_i * y_i^T * q
            let y_dot_q = y_i
                .iter()
                .zip(q.iter())
                .map(|(&y, &q_val)| y * q_val)
                .fold(A::zero(), |acc, x| acc + x);
            let beta = rho_i * y_dot_q;

            // Update q = q + (alpha_i - beta) * s_i
            let alpha_i = alphas[i];
            let coeff = alpha_i - beta;
            for (q_val, &s_val) in q.iter_mut().zip(s_i.iter()) {
                *q_val = *q_val + coeff * s_val;
            }
        }

        Ok(q)
    }

    /// Gauss-Newton Hessian approximation for least squares problems
    pub fn gauss_newton_approximation<A>(jacobian: &Array2<A>) -> Result<Array2<A>>
    where
        A: Float + ScalarOperand + Debug,
    {
        // Gauss-Newton approximation: H ≈ J^T * J
        let j_transpose = jacobian.t();
        let hessian_approx = j_transpose.dot(jacobian);
        Ok(hessian_approx)
    }
}

/// Newton's method optimizer
#[derive(Debug, Clone)]
pub struct Newton<A: Float> {
    learning_rate: A,
    regularization: A, // For numerical stability
}

impl<A: Float + ScalarOperand + Debug> Newton<A> {
    /// Create a new Newton optimizer
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            regularization: A::from(1e-6).unwrap(),
        }
    }

    /// Set regularization parameter for numerical stability
    pub fn with_regularization(mut self, regularization: A) -> Self {
        self.regularization = regularization;
        self
    }
}

impl<A: Float + ScalarOperand + Debug> SecondOrderOptimizer<A, ndarray::Ix1> for Newton<A> {
    fn step_second_order(
        &mut self,
        params: &Array1<A>,
        gradients: &Array1<A>,
        hessian_info: &HessianInfo<A, ndarray::Ix1>,
    ) -> Result<Array1<A>> {
        match hessian_info {
            HessianInfo::Diagonal(hessian_diag) => {
                if params.len() != hessian_diag.len() || params.len() != gradients.len() {
                    return Err(OptimError::DimensionMismatch(
                        "Parameter, gradient, and Hessian dimensions must match".to_string(),
                    ));
                }

                let mut update = Array1::zeros(params.len());
                for i in 0..params.len() {
                    let h_ii = hessian_diag[i] + self.regularization;
                    if h_ii.abs() > A::from(1e-12).unwrap() {
                        update[i] = gradients[i] / h_ii;
                    } else {
                        // Fall back to gradient descent if Hessian is singular
                        update[i] = gradients[i];
                    }
                }

                Ok(params - &(update * self.learning_rate))
            }
            HessianInfo::QuasiNewton {
                s_history,
                y_history,
            } => {
                // Use L-BFGS approximation
                let search_direction = hessian_approximation::lbfgs_two_loop_recursion(
                    gradients,
                    s_history,
                    y_history,
                    A::one(), // Initial Hessian scale
                )?;

                Ok(params - &(search_direction * self.learning_rate))
            }
            _ => Err(OptimError::InvalidConfig(
                "Unsupported Hessian information type for Newton method".to_string(),
            )),
        }
    }

    fn reset(&mut self) {
        // Newton method is stateless, nothing to reset
    }
}

/// Quasi-Newton L-BFGS optimizer
#[derive(Debug)]
pub struct LBFGS<A: Float, D: Dimension> {
    learning_rate: A,
    max_history: usize,
    s_history: VecDeque<Array<A, D>>,
    y_history: VecDeque<Array<A, D>>,
    previous_params: Option<Array<A, D>>,
    previous_grad: Option<Array<A, D>>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> LBFGS<A, D> {
    /// Create a new L-BFGS optimizer
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            max_history: 10,
            s_history: VecDeque::new(),
            y_history: VecDeque::new(),
            previous_params: None,
            previous_grad: None,
        }
    }

    /// Set maximum history size
    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history = max_history;
        self
    }

    /// Perform L-BFGS step
    pub fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Update history if we have previous step information
        if let (Some(prev_params), Some(prev_grad)) = (&self.previous_params, &self.previous_grad) {
            let s = params - prev_params; // Parameter difference
            let y = gradients - prev_grad; // Gradient difference

            hessian_approximation::update_lbfgs_approximation(
                &mut self.s_history,
                &mut self.y_history,
                s,
                y,
                self.max_history,
            );
        }

        // Compute search direction using two-loop recursion
        let search_direction = if self.s_history.is_empty() {
            // No history, use gradient descent
            gradients.clone()
        } else {
            hessian_approximation::lbfgs_two_loop_recursion(
                gradients,
                &self.s_history,
                &self.y_history,
                A::one(),
            )?
        };

        // Update parameters
        let new_params = params - &(search_direction * self.learning_rate);

        // Store current information for next iteration
        self.previous_params = Some(params.clone());
        self.previous_grad = Some(gradients.clone());

        Ok(new_params)
    }
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> SecondOrderOptimizer<A, D> for LBFGS<A, D> {
    fn step_second_order(
        &mut self,
        params: &Array<A, D>,
        gradients: &Array<A, D>,
        _hessian_info: &HessianInfo<A, D>, // L-BFGS maintains its own history
    ) -> Result<Array<A, D>> {
        self.step(params, gradients)
    }

    fn reset(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
        self.previous_params = None;
        self.previous_grad = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_diagonal_hessian_approximation() {
        // Test on a simple quadratic function: f(x) = x^2
        let params = Array1::from_vec(vec![1.0]);

        // Gradient function for quadratic: grad = 2*x
        let gradient_fn =
            |x: &Array1<f64>| -> Result<Array1<f64>> { Ok(Array1::from_vec(vec![2.0 * x[0]])) };

        let hessian_diag =
            hessian_approximation::diagonal_finite_difference(&params, gradient_fn, 1e-5).unwrap();

        // For quadratic function f(x) = x^2, second derivative should be 2.0
        assert_relative_eq!(hessian_diag[0], 2.0, epsilon = 1e-1);
    }

    #[test]
    fn test_lbfgs_two_loop_recursion() {
        let gradient = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut s_history = VecDeque::new();
        let mut y_history = VecDeque::new();

        // Add some history
        s_history.push_back(Array1::from_vec(vec![0.1, 0.1, 0.1]));
        y_history.push_back(Array1::from_vec(vec![0.2, 0.3, 0.4]));

        let result =
            hessian_approximation::lbfgs_two_loop_recursion(&gradient, &s_history, &y_history, 1.0)
                .unwrap();

        // Result should be different from original gradient due to curvature information
        assert_ne!(result, gradient);
        assert_eq!(result.len(), gradient.len());
    }

    #[test]
    fn test_newton_method() {
        let mut optimizer = Newton::new(0.1);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2]);
        let hessian_diag = Array1::from_vec(vec![2.0, 4.0]);

        let hessian_info = HessianInfo::Diagonal(hessian_diag);
        let new_params = optimizer
            .step_second_order(&params, &gradients, &hessian_info)
            .unwrap();

        // Verify parameters were updated
        assert!(new_params[0] < params[0]);
        assert!(new_params[1] < params[1]);
    }

    #[test]
    fn test_lbfgs_optimizer() {
        let mut optimizer = LBFGS::new(0.01).with_max_history(5);
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients1 = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let gradients2 = Array1::from_vec(vec![0.05, 0.15, 0.25]);

        // First step
        params = optimizer.step(&params, &gradients1).unwrap();

        // Second step (should use history)
        let new_params = optimizer.step(&params, &gradients2).unwrap();

        // Verify parameters were updated
        assert_ne!(new_params, params);
        assert_eq!(optimizer.s_history.len(), 1);
        assert_eq!(optimizer.y_history.len(), 1);
    }

    #[test]
    fn test_gauss_newton_approximation() {
        let jacobian = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let hessian_approx = hessian_approximation::gauss_newton_approximation(&jacobian).unwrap();

        // Should be a 2x2 matrix (J^T * J)
        assert_eq!(hessian_approx.dim(), (2, 2));

        // Verify it's positive semidefinite by checking diagonal elements are non-negative
        assert!(hessian_approx[(0, 0)] >= 0.0);
        assert!(hessian_approx[(1, 1)] >= 0.0);
    }
}
