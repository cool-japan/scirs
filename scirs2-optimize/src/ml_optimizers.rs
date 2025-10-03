//! Specialized optimizers for machine learning applications
//!
//! This module provides optimization algorithms specifically designed for
//! machine learning tasks, including regularized optimization, feature selection,
//! and deep learning optimizers with advanced features.

use crate::error::OptimizeError;
use crate::unconstrained::result::OptimizeResult;
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::seq::SliceRandom;
use std::collections::HashMap;

/// L1 regularization (Lasso) optimizer using proximal gradient descent
#[derive(Debug, Clone)]
pub struct LassoOptimizer<F: Float> {
    /// L1 regularization parameter (lambda)
    pub lambda: F,
    /// Learning rate
    pub learning_rate: F,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: F,
    /// Whether to use accelerated proximal gradient (FISTA)
    pub accelerated: bool,
}

impl<F: Float + ScalarOperand> LassoOptimizer<F> {
    /// Create a new Lasso optimizer
    pub fn new(lambda: F, learning_rate: F) -> Self {
        Self {
            lambda,
            learning_rate,
            max_iter: 1000,
            tol: F::from(1e-6).unwrap(),
            accelerated: false,
        }
    }

    /// Create a new accelerated Lasso optimizer (FISTA)
    pub fn fista(lambda: F, learning_rate: F) -> Self {
        Self {
            lambda,
            learning_rate,
            max_iter: 1000,
            tol: F::from(1e-6).unwrap(),
            accelerated: true,
        }
    }

    /// Soft thresholding operator for L1 proximal operator
    fn soft_threshold(&self, x: F, threshold: F) -> F {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            F::zero()
        }
    }

    /// Proximal operator for L1 regularization
    fn prox_l1(&self, x: &Array1<F>, step_size: F) -> Array1<F> {
        let threshold = self.lambda * step_size;
        x.mapv(|xi| self.soft_threshold(xi, threshold))
    }

    /// Optimize using proximal gradient descent
    pub fn minimize<G>(
        &self,
        mut grad_fn: G,
        x0: &Array1<F>,
    ) -> Result<OptimizeResult<F>, OptimizeError>
    where
        G: FnMut(&ArrayView1<F>) -> Array1<F>,
        F: Into<f64> + Copy,
    {
        let mut x = x0.clone();
        let mut y = x0.clone(); // For FISTA acceleration
        let mut t = F::one(); // FISTA parameter

        let _prev_loss = F::infinity();

        for iter in 0..self.max_iter {
            // Compute gradient
            let grad = if self.accelerated {
                grad_fn(&y.view())
            } else {
                grad_fn(&x.view())
            };

            // Gradient step
            let x_new = if self.accelerated {
                &y - &(&grad * self.learning_rate)
            } else {
                &x - &(&grad * self.learning_rate)
            };

            // Proximal step (L1 regularization)
            let x_prox = self.prox_l1(&x_new, self.learning_rate);

            // FISTA acceleration
            if self.accelerated {
                let t_new = (F::one() + (F::one() + F::from(4).unwrap() * t * t).sqrt())
                    / F::from(2).unwrap();
                let beta = (t - F::one()) / t_new;
                y = &x_prox + &((&x_prox - &x) * beta);
                t = t_new;
            }

            // Check convergence
            let change = (&x_prox - &x).mapv(|xi| xi.abs()).sum();
            if change < self.tol {
                return Ok(OptimizeResult {
                    x: x_prox.mapv(|v| v.into()),
                    fun: F::zero(),
                    nit: iter,
                    func_evals: iter,
                    nfev: iter,
                    success: true,
                    message: "Optimization terminated successfully.".to_string(),
                    jacobian: Some(grad.mapv(|v| v.into())),
                    hessian: None,
                });
            }

            x = x_prox;
        }

        Err(OptimizeError::ConvergenceError(
            "Maximum iterations reached".to_string(),
        ))
    }
}

/// Group Lasso optimizer for grouped feature selection
#[derive(Debug, Clone)]
pub struct GroupLassoOptimizer<F: Float> {
    /// Group Lasso regularization parameter
    pub lambda: F,
    /// Learning rate
    pub learning_rate: F,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: F,
    /// Group structure: maps feature index to group index
    pub groups: Vec<usize>,
}

impl<F: Float + ScalarOperand> GroupLassoOptimizer<F> {
    /// Create a new Group Lasso optimizer
    pub fn new(lambda: F, learning_rate: F, groups: Vec<usize>) -> Self {
        Self {
            lambda,
            learning_rate,
            max_iter: 1000,
            tol: F::from(1e-6).unwrap(),
            groups,
        }
    }

    /// Proximal operator for group L1 regularization
    fn prox_group_l1(&self, x: &Array1<F>, step_size: F) -> Array1<F> {
        let mut result = x.clone();
        let threshold = self.lambda * step_size;

        // Group variables by group index
        let mut groups_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (feature_idx, &group_idx) in self.groups.iter().enumerate() {
            groups_map.entry(group_idx).or_default().push(feature_idx);
        }

        // Apply group soft thresholding
        for (_, feature_indices) in groups_map {
            // Compute group norm
            let group_norm = feature_indices
                .iter()
                .map(|&idx| x[idx] * x[idx])
                .fold(F::zero(), |acc, x| acc + x)
                .sqrt();

            if group_norm > threshold {
                let scale = (group_norm - threshold) / group_norm;
                for &idx in &feature_indices {
                    result[idx] = result[idx] * scale;
                }
            } else {
                // Zero out the entire group
                for &idx in &feature_indices {
                    result[idx] = F::zero();
                }
            }
        }

        result
    }

    /// Optimize using proximal gradient descent
    pub fn minimize<G>(
        &self,
        mut grad_fn: G,
        x0: &Array1<F>,
    ) -> Result<OptimizeResult<F>, OptimizeError>
    where
        G: FnMut(&ArrayView1<F>) -> Array1<F>,
        F: Into<f64> + Copy,
    {
        let mut x = x0.clone();

        for iter in 0..self.max_iter {
            // Compute gradient
            let grad = grad_fn(&x.view());

            // Gradient step
            let x_new = &x - &(&grad * self.learning_rate);

            // Proximal step (group L1 regularization)
            let x_prox = self.prox_group_l1(&x_new, self.learning_rate);

            // Check convergence
            let change = (&x_prox - &x).mapv(|xi| xi.abs()).sum();
            if change < self.tol {
                return Ok(OptimizeResult {
                    x: x_prox.mapv(|v| v.into()),
                    fun: F::zero(),
                    nit: iter,
                    func_evals: iter,
                    nfev: iter,
                    success: true,
                    message: "Optimization terminated successfully.".to_string(),
                    jacobian: Some(grad.mapv(|v| v.into())),
                    hessian: None,
                });
            }

            x = x_prox;
        }

        Err(OptimizeError::ConvergenceError(
            "Maximum iterations reached".to_string(),
        ))
    }
}

/// Elastic Net optimizer (L1 + L2 regularization)
#[derive(Debug, Clone)]
pub struct ElasticNetOptimizer<F: Float> {
    /// L1 regularization parameter
    pub lambda1: F,
    /// L2 regularization parameter  
    pub lambda2: F,
    /// Learning rate
    pub learning_rate: F,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: F,
}

impl<F: Float + ScalarOperand> ElasticNetOptimizer<F> {
    /// Create a new Elastic Net optimizer
    pub fn new(lambda1: F, lambda2: F, learning_rate: F) -> Self {
        Self {
            lambda1,
            lambda2,
            learning_rate,
            max_iter: 1000,
            tol: F::from(1e-6).unwrap(),
        }
    }

    /// Proximal operator for elastic net regularization
    fn prox_elastic_net(&self, x: &Array1<F>, step_size: F) -> Array1<F> {
        let l1_threshold = self.lambda1 * step_size;
        let l2_factor = F::one() / (F::one() + self.lambda2 * step_size);

        x.mapv(|xi| {
            let soft_thresh = if xi > l1_threshold {
                xi - l1_threshold
            } else if xi < -l1_threshold {
                xi + l1_threshold
            } else {
                F::zero()
            };
            soft_thresh * l2_factor
        })
    }

    /// Optimize using proximal gradient descent
    pub fn minimize<G>(
        &self,
        mut grad_fn: G,
        x0: &Array1<F>,
    ) -> Result<OptimizeResult<F>, OptimizeError>
    where
        G: FnMut(&ArrayView1<F>) -> Array1<F>,
        F: Into<f64> + Copy,
    {
        let mut x = x0.clone();

        for iter in 0..self.max_iter {
            // Compute gradient
            let grad = grad_fn(&x.view());

            // Gradient step
            let x_new = &x - &(&grad * self.learning_rate);

            // Proximal step (elastic net regularization)
            let x_prox = self.prox_elastic_net(&x_new, self.learning_rate);

            // Check convergence
            let change = (&x_prox - &x).mapv(|xi| xi.abs()).sum();
            if change < self.tol {
                return Ok(OptimizeResult {
                    x: x_prox.mapv(|v| v.into()),
                    fun: F::zero(),
                    nit: iter,
                    func_evals: iter,
                    nfev: iter,
                    success: true,
                    message: "Optimization terminated successfully.".to_string(),
                    jacobian: Some(grad.mapv(|v| v.into())),
                    hessian: None,
                });
            }

            x = x_prox;
        }

        Err(OptimizeError::ConvergenceError(
            "Maximum iterations reached".to_string(),
        ))
    }
}

/// ADMM (Alternating Direction Method of Multipliers) optimizer for general constrained problems
#[derive(Debug, Clone)]
pub struct ADMMOptimizer<F: Float> {
    /// Penalty parameter
    pub rho: F,
    /// Primary tolerance
    pub eps_pri: F,
    /// Dual tolerance
    pub eps_dual: F,
    /// Maximum number of iterations
    pub max_iter: usize,
}

impl<F: Float + ScalarOperand> ADMMOptimizer<F> {
    /// Create a new ADMM optimizer
    pub fn new(rho: F) -> Self {
        Self {
            rho,
            eps_pri: F::from(1e-3).unwrap(),
            eps_dual: F::from(1e-3).unwrap(),
            max_iter: 1000,
        }
    }

    /// Solve Lasso using ADMM
    pub fn solve_lasso<LossGrad, Data>(
        &self,
        loss_grad: LossGrad,
        lambda: F,
        x0: &Array1<F>,
        data: &Data,
    ) -> Result<OptimizeResult<F>, OptimizeError>
    where
        LossGrad: Fn(&ArrayView1<F>, &Data) -> Array1<F>,
        Data: Clone,
        F: Into<f64> + Copy,
    {
        let n = x0.len();
        let mut x = x0.clone();
        let mut z = Array1::zeros(n);
        let mut u = Array1::zeros(n); // Scaled dual variable

        for iter in 0..self.max_iter {
            // x-update: minimize loss + (rho/2)||x - z + u||^2
            let grad_loss = loss_grad(&x.view(), data);
            let grad_augmented = &grad_loss + &((&x - &z + &u) * self.rho);

            // Adaptive learning rate for better convergence
            let grad_norm = grad_augmented.mapv(|g| g * g).sum().sqrt();
            let lr = if grad_norm > F::epsilon() {
                F::one() / (F::one() + self.rho)
            } else {
                F::from(0.1).unwrap()
            };
            x = &x - &(&grad_augmented * lr);

            // z-update: soft thresholding
            let z_old = z.clone();
            let threshold = lambda / self.rho;
            z = (&x + &u).mapv(|xi| {
                if xi > threshold {
                    xi - threshold
                } else if xi < -threshold {
                    xi + threshold
                } else {
                    F::zero()
                }
            });

            // u-update: dual variable update
            u = &u + &x - &z;

            // Check convergence
            let r_norm = (&x - &z).mapv(|xi| xi * xi).sum().sqrt(); // Primal residual
            let s_norm = ((&z - &z_old) * self.rho).mapv(|xi| xi * xi).sum().sqrt(); // Dual residual

            let x_norm = x.mapv(|xi| xi * xi).sum().sqrt();
            let z_norm = z.mapv(|xi| xi * xi).sum().sqrt();
            let u_norm = u.mapv(|xi| xi * xi).sum().sqrt();

            let eps_pri_thresh =
                self.eps_pri * (F::sqrt(F::from(n).unwrap()) + F::max(x_norm, z_norm));
            let eps_dual_thresh = self.eps_dual * F::sqrt(F::from(n).unwrap()) * self.rho * u_norm;

            if r_norm < eps_pri_thresh && s_norm < eps_dual_thresh {
                return Ok(OptimizeResult {
                    x: x.mapv(|v| v.into()),
                    fun: F::zero(),
                    nit: iter,
                    func_evals: iter,
                    nfev: iter,
                    success: true,
                    message: "ADMM converged successfully.".to_string(),
                    jacobian: Some(grad_loss.mapv(|v| v.into())),
                    hessian: None,
                });
            }
        }

        Err(OptimizeError::ConvergenceError(
            "Maximum iterations reached".to_string(),
        ))
    }
}

/// Coordinate Descent optimizer for separable problems
#[derive(Debug, Clone)]
pub struct CoordinateDescentOptimizer<F: Float> {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: F,
    /// Whether to use random coordinate selection
    pub random: bool,
}

impl<F: Float + ScalarOperand> Default for CoordinateDescentOptimizer<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: F::from(1e-6).unwrap(),
            random: false,
        }
    }
}

impl<F: Float + ScalarOperand> CoordinateDescentOptimizer<F> {
    /// Create a new coordinate descent optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new random coordinate descent optimizer
    pub fn random() -> Self {
        Self {
            max_iter: 1000,
            tol: F::from(1e-6).unwrap(),
            random: true,
        }
    }

    /// Minimize using coordinate descent
    pub fn minimize<Obj, Grad1D>(
        &self,
        obj_fn: Obj,
        grad_1d_fn: Grad1D,
        x0: &Array1<F>,
    ) -> Result<OptimizeResult<F>, OptimizeError>
    where
        Obj: Fn(&ArrayView1<F>) -> F,
        Grad1D: Fn(&ArrayView1<F>, usize) -> F, // Partial derivative w.r.t. coordinate i
        F: Into<f64> + Copy,
    {
        let mut x = x0.clone();
        let n = x.len();

        for iter in 0..self.max_iter {
            let x_old = x.clone();

            // Choose coordinate order
            let coords: Vec<usize> = if self.random {
                use scirs2_core::random::{rng, seq::SliceRandom};
                let mut coords: Vec<usize> = (0..n).collect();
                coords.shuffle(&mut scirs2_core::random::rng());
                coords
            } else {
                (0..n).collect()
            };

            // Update each coordinate
            for &i in &coords {
                let grad_i = grad_1d_fn(&x.view(), i);

                // Simple gradient step for coordinate i
                let lr = F::from(0.01).unwrap();
                x[i] = x[i] - lr * grad_i;
            }

            // Check convergence
            let change = (&x - &x_old).mapv(|xi| xi.abs()).sum();
            if change < self.tol {
                let final_obj = obj_fn(&x.view());
                return Ok(OptimizeResult {
                    x: x.mapv(|v| v.into()),
                    fun: final_obj,
                    nit: iter,
                    func_evals: iter * n,
                    nfev: iter * n,
                    success: true,
                    message: "Coordinate descent converged successfully.".to_string(),
                    jacobian: Some(Array1::zeros(n)), // Could compute full gradient here
                    hessian: None,
                });
            }
        }

        Err(OptimizeError::ConvergenceError(
            "Maximum iterations reached".to_string(),
        ))
    }
}

/// Convenience functions for common ML optimization problems
pub mod ml_problems {
    use super::*;

    /// Solve Lasso regression: min ||Ax - b||^2 + lambda ||x||_1
    pub fn lasso_regression<F: Float + ScalarOperand + Into<f64> + Copy>(
        a: &ArrayView2<F>,
        b: &ArrayView1<F>,
        lambda: F,
        learning_rate: Option<F>,
    ) -> Result<OptimizeResult<F>, OptimizeError> {
        let lr = learning_rate.unwrap_or_else(|| F::from(0.01).unwrap());
        let optimizer = LassoOptimizer::new(lambda, lr);

        let n = a.ncols();
        let x0 = Array1::zeros(n);

        let grad_fn = |x: &ArrayView1<F>| -> Array1<F> {
            let residual = a.dot(x) - b;
            a.t().dot(&residual) * F::from(2).unwrap()
        };

        optimizer.minimize(grad_fn, &x0)
    }

    /// Solve Group Lasso regression
    pub fn group_lasso_regression<F: Float + ScalarOperand + Into<f64> + Copy>(
        a: &ArrayView2<F>,
        b: &ArrayView1<F>,
        lambda: F,
        groups: Vec<usize>,
        learning_rate: Option<F>,
    ) -> Result<OptimizeResult<F>, OptimizeError> {
        let lr = learning_rate.unwrap_or_else(|| F::from(0.01).unwrap());
        let optimizer = GroupLassoOptimizer::new(lambda, lr, groups);

        let n = a.ncols();
        let x0 = Array1::zeros(n);

        let grad_fn = |x: &ArrayView1<F>| -> Array1<F> {
            let residual = a.dot(x) - b;
            a.t().dot(&residual) * F::from(2).unwrap()
        };

        optimizer.minimize(grad_fn, &x0)
    }

    /// Solve Elastic Net regression: min ||Ax - b||^2 + lambda1 ||x||_1 + lambda2 ||x||^2
    pub fn elastic_net_regression<F: Float + ScalarOperand + Into<f64> + Copy>(
        a: &ArrayView2<F>,
        b: &ArrayView1<F>,
        lambda1: F,
        lambda2: F,
        learning_rate: Option<F>,
    ) -> Result<OptimizeResult<F>, OptimizeError> {
        let lr = learning_rate.unwrap_or_else(|| F::from(0.01).unwrap());
        let optimizer = ElasticNetOptimizer::new(lambda1, lambda2, lr);

        let n = a.ncols();
        let x0 = Array1::zeros(n);

        let grad_fn = |x: &ArrayView1<F>| -> Array1<F> {
            let residual = a.dot(x) - b;
            a.t().dot(&residual) * F::from(2).unwrap()
        };

        optimizer.minimize(grad_fn, &x0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_lasso_soft_threshold() {
        let optimizer = LassoOptimizer::new(1.0, 0.1);

        // Test soft thresholding
        assert_abs_diff_eq!(optimizer.soft_threshold(2.0, 1.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(optimizer.soft_threshold(-2.0, 1.0), -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(optimizer.soft_threshold(0.5, 1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lasso_prox_operator() {
        let optimizer = LassoOptimizer::new(1.0, 0.1);
        let x = array![2.0, -2.0, 0.5, -0.5];
        let result = optimizer.prox_l1(&x, 1.0);

        assert_abs_diff_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_elastic_net_prox_operator() {
        let optimizer = ElasticNetOptimizer::new(1.0, 1.0, 0.1);
        let x = array![3.0, -3.0, 0.5];
        let result = optimizer.prox_elastic_net(&x, 1.0);

        // With L2 regularization, should shrink more than pure L1
        assert!(result[0] > 0.0 && result[0] < 2.0);
        assert!(result[1] < 0.0 && result[1] > -2.0);
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_group_lasso_simple() {
        let optimizer = GroupLassoOptimizer::new(1.0, 0.1, vec![0, 0, 1, 1]);
        let x = array![1.0, 1.0, 0.1, 0.1];
        let result = optimizer.prox_group_l1(&x, 1.0);

        // First group (0,1) should be shrunk but not zeroed
        assert!(result[0] > 0.0 && result[0] < 1.0);
        assert!(result[1] > 0.0 && result[1] < 1.0);

        // Second group (2,3) should be zeroed (small norm)
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_coordinate_descent_optimizer() {
        let optimizer = CoordinateDescentOptimizer::new();

        // Simple quadratic: f(x) = x1^2 + x2^2
        let obj_fn = |x: &ArrayView1<f64>| x.mapv(|xi| xi * xi).sum();
        let grad_1d_fn = |x: &ArrayView1<f64>, i: usize| 2.0 * x[i];

        let x0 = array![1.0, 1.0];
        let result = optimizer.minimize(obj_fn, grad_1d_fn, &x0).unwrap();

        // Should converge to origin
        assert!(result.x[0].abs() < 0.1);
        assert!(result.x[1].abs() < 0.1);
        assert!(result.success);
    }

    #[test]
    fn test_admm_optimizer() {
        let optimizer = ADMMOptimizer::new(1.0);

        // Simple test: minimize x^2 + lambda|x|
        let loss_grad = |x: &ArrayView1<f64>, _data: &()| x.mapv(|xi| 2.0 * xi);

        let x0 = array![2.0];
        let lambda = 1.0;
        let result = optimizer.solve_lasso(loss_grad, lambda, &x0, &()).unwrap();

        // Should converge to a sparse solution
        assert!(result.x[0].abs() < 1.0); // Should be shrunk
        assert!(result.success);
    }

    #[test]
    fn test_ml_lasso_regression() {
        // Simple 2D problem: y = x1 + noise, x2 should be zero
        let a = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];
        let b = array![1.0, 1.1, 0.9];
        let lambda = 0.1;

        let result = ml_problems::lasso_regression(&a.view(), &b.view(), lambda, None).unwrap();

        // x1 should be close to 1, x2 should be close to 0
        assert!(result.x[0] > 0.5);
        assert!(result.x[1].abs() < 0.1);
    }
}
