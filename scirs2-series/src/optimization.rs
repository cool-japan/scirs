//! Optimization algorithms for time series models
//!
//! Implements numerical optimization methods for parameter estimation

use ndarray::{Array1, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};

/// Optimization options
#[derive(Debug, Clone)]
pub struct OptimizationOptions<F> {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: F,
    /// Initial step size
    pub initial_step: F,
    /// Line search parameter alpha
    pub line_search_alpha: F,
    /// Line search parameter beta
    pub line_search_beta: F,
    /// Gradient tolerance
    pub grad_tolerance: F,
}

impl<F: Float + FromPrimitive> Default for OptimizationOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tolerance: F::from(1e-8).unwrap(),
            initial_step: F::from(0.1).unwrap(),
            line_search_alpha: F::from(0.3).unwrap(),
            line_search_beta: F::from(0.8).unwrap(),
            grad_tolerance: F::from(1e-6).unwrap(),
        }
    }
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult<F> {
    /// Optimal parameters
    pub x: Array1<F>,
    /// Function value at optimum
    pub fval: F,
    /// Number of iterations
    pub iterations: usize,
    /// Convergence flag
    pub converged: bool,
    /// Final gradient norm
    pub grad_norm: F,
}

/// L-BFGS optimizer for unconstrained optimization
pub struct LBFGSOptimizer<F> {
    options: OptimizationOptions<F>,
    memory_size: usize,
    s_history: Vec<Array1<F>>,
    y_history: Vec<Array1<F>>,
    rho_history: Vec<F>,
}

impl<F> LBFGSOptimizer<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    /// Create a new L-BFGS optimizer
    pub fn new(options: OptimizationOptions<F>) -> Self {
        Self {
            options,
            memory_size: 10,
            s_history: Vec::new(),
            y_history: Vec::new(),
            rho_history: Vec::new(),
        }
    }

    /// Optimize a function
    pub fn optimize<Func, Grad>(
        &mut self,
        f: Func,
        grad: Grad,
        x0: &Array1<F>,
    ) -> Result<OptimizationResult<F>>
    where
        Func: Fn(&Array1<F>) -> F,
        Grad: Fn(&Array1<F>) -> Array1<F>,
    {
        let mut x = x0.clone();
        let mut g = grad(&x);
        let mut fval = f(&x);

        self.s_history.clear();
        self.y_history.clear();
        self.rho_history.clear();

        for iter in 0..self.options.max_iter {
            // Check gradient convergence
            let grad_norm = g.dot(&g).sqrt();
            if grad_norm < self.options.grad_tolerance {
                return Ok(OptimizationResult {
                    x,
                    fval,
                    iterations: iter,
                    converged: true,
                    grad_norm,
                });
            }

            // Compute search direction
            let d = self.compute_direction(&g)?;

            // Line search
            let alpha = line_search_armijo(&x, &d, &f, &grad, &self.options)?;

            // Update x
            let x_new = &x + &(&d * alpha);
            let g_new = grad(&x_new);
            let fval_new = f(&x_new);

            // Update history
            let s = &x_new - &x;
            let y = &g_new - &g;
            let rho = F::one() / y.dot(&s);

            if self.s_history.len() >= self.memory_size {
                self.s_history.remove(0);
                self.y_history.remove(0);
                self.rho_history.remove(0);
            }

            self.s_history.push(s);
            self.y_history.push(y);
            self.rho_history.push(rho);

            // Check convergence
            if (fval - fval_new).abs() < self.options.tolerance {
                return Ok(OptimizationResult {
                    x: x_new,
                    fval: fval_new,
                    iterations: iter + 1,
                    converged: true,
                    grad_norm,
                });
            }

            x = x_new;
            g = g_new;
            fval = fval_new;
        }

        Ok(OptimizationResult {
            x,
            fval,
            iterations: self.options.max_iter,
            converged: false,
            grad_norm: g.dot(&g).sqrt(),
        })
    }

    /// Compute search direction using L-BFGS
    fn compute_direction(&self, g: &Array1<F>) -> Result<Array1<F>> {
        if self.s_history.is_empty() {
            return Ok(g.mapv(|x| -x));
        }

        let mut q = g.clone();
        let mut alpha = vec![F::zero(); self.s_history.len()];

        // First loop
        for i in (0..self.s_history.len()).rev() {
            alpha[i] = self.rho_history[i] * self.s_history[i].dot(&q);
            q = &q - &(&self.y_history[i] * alpha[i]);
        }

        // Initial Hessian approximation
        let gamma = self
            .s_history
            .last()
            .unwrap()
            .dot(self.y_history.last().unwrap())
            / self
                .y_history
                .last()
                .unwrap()
                .dot(self.y_history.last().unwrap());
        let mut r = &q * gamma;

        // Second loop
        for (i, alpha_val) in alpha.iter().enumerate() {
            let beta = self.rho_history[i] * self.y_history[i].dot(&r);
            r = &r + &(&self.s_history[i] * (*alpha_val - beta));
        }

        Ok(r.mapv(|x| -x))
    }
}

/// BFGS optimizer
pub struct BFGSOptimizer<F> {
    options: OptimizationOptions<F>,
    h_inv: Option<Array2<F>>,
}

use ndarray::Array2;

impl<F> BFGSOptimizer<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    /// Create a new BFGS optimizer
    pub fn new(options: OptimizationOptions<F>) -> Self {
        Self {
            options,
            h_inv: None,
        }
    }

    /// Optimize a function
    pub fn optimize<Func, Grad>(
        &mut self,
        f: Func,
        grad: Grad,
        x0: &Array1<F>,
    ) -> Result<OptimizationResult<F>>
    where
        Func: Fn(&Array1<F>) -> F,
        Grad: Fn(&Array1<F>) -> Array1<F>,
    {
        let n = x0.len();
        let mut x = x0.clone();
        let mut g = grad(&x);
        let mut fval = f(&x);

        // Initialize inverse Hessian approximation
        if self.h_inv.is_none() {
            self.h_inv = Some(Array2::eye(n));
        }
        let h_inv = self.h_inv.as_mut().unwrap();

        for iter in 0..self.options.max_iter {
            // Check gradient convergence
            let grad_norm = g.dot(&g).sqrt();
            if grad_norm < self.options.grad_tolerance {
                return Ok(OptimizationResult {
                    x,
                    fval,
                    iterations: iter,
                    converged: true,
                    grad_norm,
                });
            }

            // Compute search direction
            let d = -h_inv.dot(&g);

            // Line search
            let alpha = line_search_armijo(&x, &d, &f, &grad, &self.options)?;

            // Update x
            let x_new = &x + &(&d * alpha);
            let g_new = grad(&x_new);
            let fval_new = f(&x_new);

            // Update inverse Hessian
            let s = &x_new - &x;
            let y = &g_new - &g;
            let sy = s.dot(&y);

            if sy > F::from(1e-8).unwrap() {
                let rho = F::one() / sy;
                let sy_outer = s
                    .clone()
                    .insert_axis(ndarray::Axis(1))
                    .dot(&y.clone().insert_axis(ndarray::Axis(0)));
                let ys_outer = y
                    .clone()
                    .insert_axis(ndarray::Axis(1))
                    .dot(&s.clone().insert_axis(ndarray::Axis(0)));
                let ss_outer = s
                    .clone()
                    .insert_axis(ndarray::Axis(1))
                    .dot(&s.clone().insert_axis(ndarray::Axis(0)));

                let i_minus_rho_sy = Array2::eye(n) - &sy_outer * rho;
                let i_minus_rho_ys = Array2::eye(n) - &ys_outer * rho;

                *h_inv = i_minus_rho_sy.dot(h_inv).dot(&i_minus_rho_ys) + &ss_outer * rho;
            }

            // Check convergence
            if (fval - fval_new).abs() < self.options.tolerance {
                return Ok(OptimizationResult {
                    x: x_new,
                    fval: fval_new,
                    iterations: iter + 1,
                    converged: true,
                    grad_norm,
                });
            }

            x = x_new;
            g = g_new;
            fval = fval_new;
        }

        Ok(OptimizationResult {
            x,
            fval,
            iterations: self.options.max_iter,
            converged: false,
            grad_norm: g.dot(&g).sqrt(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lbfgs_optimizer() {
        // Test minimizing a simple quadratic function: f(x) = x^T x
        let f = |x: &Array1<f64>| x.dot(x);
        let grad = |x: &Array1<f64>| 2.0 * x;

        let mut optimizer = LBFGSOptimizer::new(OptimizationOptions::default());
        let x0 = array![1.0, 2.0, 3.0];
        let result = optimizer.optimize(f, grad, &x0).unwrap();

        assert!(result.converged);
        assert!(result.fval < 1e-6);
        assert!(result.grad_norm < 1e-6);
    }

    #[test]
    fn test_rosenbrock_function() {
        // Test on Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        let f = |x: &Array1<f64>| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };

        let grad = |x: &Array1<f64>| {
            let dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            let dy = 200.0 * (x[1] - x[0] * x[0]);
            array![dx, dy]
        };

        let mut optimizer = LBFGSOptimizer::new(OptimizationOptions {
            max_iter: 1000,
            tolerance: 1e-8,
            grad_tolerance: 1e-6,
            ..Default::default()
        });

        let x0 = array![-1.0, 1.0];
        let result = optimizer.optimize(f, grad, &x0).unwrap();

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 0.01);
        assert!((result.x[1] - 1.0).abs() < 0.01);
    }
}

/// Armijo line search (standalone)
fn line_search_armijo<F, Func, Grad>(
    x: &Array1<F>,
    d: &Array1<F>,
    f: &Func,
    grad: &Grad,
    options: &OptimizationOptions<F>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
    Func: Fn(&Array1<F>) -> F,
    Grad: Fn(&Array1<F>) -> Array1<F>,
{
    let mut alpha = F::one();
    let f0 = f(x);
    let g0 = grad(x);
    let dg0 = g0.dot(d);

    if dg0 > F::zero() {
        return Err(TimeSeriesError::ComputationError(
            "Invalid search direction".to_string(),
        ));
    }

    while alpha > F::from(1e-10).unwrap() {
        let x_new = x + &(d * alpha);
        let f_new = f(&x_new);

        if f_new <= f0 + options.line_search_alpha * alpha * dg0 {
            return Ok(alpha);
        }

        alpha = alpha * options.line_search_beta;
    }

    Ok(F::from(1e-10).unwrap())
}
