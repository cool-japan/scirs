//! L-BFGS optimizer implementation
//!
//! Based on the Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm.

use ndarray::{Array, Array1, Dimension, ScalarOperand};
use num_traits::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// L-BFGS optimizer
///
/// Implements the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm.
/// This is a quasi-Newton method that approximates the Hessian inverse using a limited
/// amount of memory by storing only a few vectors from previous iterations.
///
/// # Examples
///
/// ```no_run
/// use ndarray::Array1;
/// use scirs2_optim::optimizers::{LBFGS, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create an L-BFGS optimizer
/// let mut optimizer = LBFGS::new(1.0);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LBFGS<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// History size (number of vectors to store)
    history_size: usize,
    /// Tolerance for gradient norm
    tolerance_grad: A,
    /// Wolfe line search parameter c1
    #[allow(dead_code)]
    c1: A,
    /// Wolfe line search parameter c2
    #[allow(dead_code)]
    c2: A,
    /// Maximum number of line search iterations
    #[allow(dead_code)]
    max_ls: usize,
    /// History of gradient differences (y = grad_new - grad_old)
    old_dirs: VecDeque<Array1<A>>,
    /// History of step vectors (s = params_new - params_old)
    old_stps: VecDeque<Array1<A>>,
    /// History of 1/(y·s) values
    ro: VecDeque<A>,
    /// Previous gradient
    prev_grad: Option<Array1<A>>,
    /// Initial Hessian diagonal value
    h_diag: A,
    /// Step counter
    n_iter: usize,
    /// Temporary alpha values for two-loop recursion
    alpha: Vec<A>,
}

impl<A: Float + ScalarOperand + Debug> LBFGS<A> {
    /// Creates a new L-BFGS optimizer with the given learning rate
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self::new_with_config(
            learning_rate,
            100,                    // history_size
            A::from(1e-7).unwrap(), // tolerance_grad
            A::from(1e-4).unwrap(), // c1
            A::from(0.9).unwrap(),  // c2
            25,                     // max_ls
        )
    }

    /// Creates a new L-BFGS optimizer with full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `history_size` - Number of past gradients/steps to store (default: 100)
    /// * `tolerance_grad` - Gradient norm tolerance for convergence (default: 1e-7)
    /// * `c1` - Wolfe line search parameter for Armijo condition (default: 1e-4)
    /// * `c2` - Wolfe line search parameter for curvature condition (default: 0.9)
    /// * `max_ls` - Maximum line search iterations (default: 25)
    pub fn new_with_config(
        learning_rate: A,
        history_size: usize,
        tolerance_grad: A,
        c1: A,
        c2: A,
        max_ls: usize,
    ) -> Self {
        Self {
            learning_rate,
            history_size,
            tolerance_grad,
            c1,
            c2,
            max_ls,
            old_dirs: VecDeque::with_capacity(history_size),
            old_stps: VecDeque::with_capacity(history_size),
            ro: VecDeque::with_capacity(history_size),
            prev_grad: None,
            h_diag: A::one(),
            n_iter: 0,
            alpha: vec![A::zero(); history_size],
        }
    }

    /// Gets the current learning rate
    pub fn learning_rate(&self) -> A {
        self.learning_rate
    }

    /// Sets the learning rate
    pub fn set_lr(&mut self, lr: A) {
        self.learning_rate = lr;
    }

    /// Resets the internal state of the optimizer
    pub fn reset(&mut self) {
        self.old_dirs.clear();
        self.old_stps.clear();
        self.ro.clear();
        self.prev_grad = None;
        self.h_diag = A::one();
        self.n_iter = 0;
        self.alpha.fill(A::zero());
    }

    /// Performs the two-loop recursion to compute the search direction
    fn compute_direction(&mut self, gradient: &Array1<A>) -> Array1<A> {
        // If first iteration, use negative gradient
        if self.n_iter == 0 {
            return gradient.mapv(|x| -x);
        }

        let num_old = self.old_dirs.len();

        // First loop: compute alpha values and initial direction
        let mut q = gradient.mapv(|x| -x);

        for i in (0..num_old).rev() {
            self.alpha[i] = self.old_stps[i].dot(&q) * self.ro[i];
            q = &q - &self.old_dirs[i] * self.alpha[i];
        }

        // Scale by initial Hessian
        let mut r = q * self.h_diag;

        // Second loop: compute final direction
        for i in 0..num_old {
            let beta = self.old_dirs[i].dot(&r) * self.ro[i];
            r = &r + &self.old_stps[i] * (self.alpha[i] - beta);
        }

        r
    }

    /// Updates the history with new gradient and step information
    fn update_history(&mut self, y: Array1<A>, s: Array1<A>) {
        let ys = y.dot(&s);

        // Only update if y·s is positive (ensures positive definiteness)
        if ys > A::from(1e-10).unwrap() {
            // Remove oldest entries if at capacity
            if self.old_dirs.len() >= self.history_size {
                self.old_dirs.pop_front();
                self.old_stps.pop_front();
                self.ro.pop_front();
            }

            // Add new entries
            self.old_dirs.push_back(y.clone());
            self.old_stps.push_back(s);
            self.ro.push_back(A::one() / ys);

            // Update initial Hessian approximation
            let yy = y.dot(&y);
            if yy > A::zero() {
                self.h_diag = ys / yy;
            }
        }
    }
}

impl<A, D> Optimizer<A, D> for LBFGS<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Convert to 1D for computation
        let params_flat = params
            .to_owned()
            .into_shape_with_order(params.len())
            .unwrap();
        let gradients_flat = gradients
            .to_owned()
            .into_shape_with_order(gradients.len())
            .unwrap();

        // Check convergence
        let grad_norm = gradients_flat.dot(&gradients_flat).sqrt();
        if grad_norm <= self.tolerance_grad {
            return Ok(params.clone());
        }

        // Update history if we have previous gradient (before computing new direction)
        if let Some(prev_grad) = self.prev_grad.clone() {
            let y = &gradients_flat - &prev_grad;

            // Only update if we've moved in parameter space
            if self.n_iter > 0 {
                // Compute actual step taken in the previous iteration
                let direction = self.compute_direction(&prev_grad);
                let step_size = if self.n_iter == 1 {
                    self.learning_rate / (A::one() + grad_norm)
                } else {
                    self.learning_rate
                };
                let s = direction * step_size;
                self.update_history(y, s);
            }
        }

        // Compute search direction using current gradient
        let direction = self.compute_direction(&gradients_flat);

        // Simple line search with fixed step size (no backtracking for now)
        let step_size = if self.n_iter == 0 {
            // First iteration: use smaller step
            self.learning_rate / (A::one() + grad_norm)
        } else {
            self.learning_rate
        };

        // Update parameters
        let new_params_flat = &params_flat + &(&direction * step_size);

        // Store current gradient for next iteration
        self.prev_grad = Some(gradients_flat.clone());
        self.n_iter += 1;

        // Reshape back to original dimensions
        let new_params = new_params_flat
            .into_shape_with_order(params.raw_dim())
            .unwrap();

        Ok(new_params)
    }

    fn get_learning_rate(&self) -> A {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.learning_rate = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_lbfgs_basic_creation() {
        let optimizer: LBFGS<f64> = LBFGS::new(1.0);
        assert_abs_diff_eq!(optimizer.learning_rate(), 1.0);
        assert_eq!(optimizer.history_size, 100);
        assert_abs_diff_eq!(optimizer.tolerance_grad, 1e-7);
    }

    #[test]
    fn test_lbfgs_convergence() {
        let mut optimizer: LBFGS<f64> = LBFGS::new(0.1);

        // Minimize f(x) = x^2
        let mut params = Array1::from_vec(vec![10.0]);

        for _ in 0..50 {
            let gradients = Array1::from_vec(vec![2.0 * params[0]]);
            params = optimizer.step(&params, &gradients).unwrap();
        }

        // Should converge close to 0
        assert!(params[0].abs() < 0.1);
    }

    #[test]
    fn test_lbfgs_2d() {
        let mut optimizer: LBFGS<f64> = LBFGS::new(0.1);

        // Minimize f(x,y) = x^2 + y^2
        let mut params = Array1::from_vec(vec![5.0, 3.0]);

        for _ in 0..50 {
            let gradients = Array1::from_vec(vec![2.0 * params[0], 2.0 * params[1]]);
            params = optimizer.step(&params, &gradients).unwrap();
        }

        // Should converge close to (0, 0)
        assert!(params[0].abs() < 0.1);
        assert!(params[1].abs() < 0.1);
    }

    #[test]
    fn test_lbfgs_reset() {
        let mut optimizer: LBFGS<f64> = LBFGS::new(0.1);

        // Perform some steps
        let mut params = Array1::from_vec(vec![1.0]);
        let gradients = Array1::from_vec(vec![2.0]);
        params = optimizer.step(&params, &gradients).unwrap();

        // Need one more step to actually update history
        let gradients2 = Array1::from_vec(vec![1.5]);
        params = optimizer.step(&params, &gradients2).unwrap();

        // Third step to populate history
        let gradients3 = Array1::from_vec(vec![1.0]);
        let _ = optimizer.step(&params, &gradients3).unwrap();

        // Verify state exists
        assert!(!optimizer.old_dirs.is_empty());
        assert!(optimizer.n_iter > 0);

        // Reset
        optimizer.reset();

        // Verify state is cleared
        assert!(optimizer.old_dirs.is_empty());
        assert!(optimizer.old_stps.is_empty());
        assert!(optimizer.ro.is_empty());
        assert!(optimizer.prev_grad.is_none());
        assert_eq!(optimizer.n_iter, 0);
    }
}
