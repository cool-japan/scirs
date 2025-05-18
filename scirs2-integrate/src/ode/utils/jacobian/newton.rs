//! Newton solver for nonlinear equations
//!
//! This module provides Newton-type solvers for nonlinear equations
//! that arise in implicit ODE solvers. It includes standard Newton's method,
//! modified Newton's method, and inexact Newton variants.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::utils::jacobian::JacobianManager;
use crate::ode::utils::linear_solvers::{auto_solve_linear_system, LinearSolverType};
use ndarray::Array1;

/// Newton solver parameters
#[derive(Debug, Clone)]
pub struct NewtonParameters<F: IntegrateFloat> {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Absolute tolerance for convergence
    pub abs_tolerance: F,
    /// Relative tolerance for convergence
    pub rel_tolerance: F,
    /// Jacobian update frequency (1 = every iteration)
    pub jacobian_update_freq: usize,
    /// Step size damping factor (1.0 = full steps)
    pub damping_factor: F,
    /// Minimum allowed step size factor
    pub min_damping: F,
    /// Reuse Jacobian from previous solve
    pub reuse_jacobian: bool,
    /// Force Jacobian update on first iteration
    pub force_jacobian_init: bool,
}

impl<F: IntegrateFloat> Default for NewtonParameters<F> {
    fn default() -> Self {
        NewtonParameters {
            max_iterations: 10,
            abs_tolerance: F::from_f64(1e-10).unwrap(),
            rel_tolerance: F::from_f64(1e-8).unwrap(),
            jacobian_update_freq: 1,
            damping_factor: F::one(),
            min_damping: F::from_f64(0.1).unwrap(),
            reuse_jacobian: true,
            force_jacobian_init: false,
        }
    }
}

/// Result of a Newton solve
#[derive(Debug, Clone)]
pub struct NewtonResult<F: IntegrateFloat> {
    /// Solution vector
    pub solution: Array1<F>,
    /// Residual at solution
    pub residual: Array1<F>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Convergence flag
    pub converged: bool,
    /// Error estimate
    pub error_estimate: F,
    /// Number of function evaluations
    pub func_evals: usize,
    /// Number of Jacobian evaluations
    pub jac_evals: usize,
    /// Number of linear solves
    pub linear_solves: usize,
}

/// Solve a nonlinear system F(x) = 0 using Newton's method
///
/// This version uses the improved Jacobian handling for performance
#[allow(clippy::explicit_counter_loop)]
pub fn newton_solve<F, Func>(
    f: Func,
    x0: Array1<F>,
    jac_manager: &mut JacobianManager<F>,
    params: NewtonParameters<F>,
) -> IntegrateResult<NewtonResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(&Array1<F>) -> Array1<F>,
{
    let _n = x0.len();
    let mut x = x0.clone();
    let mut residual = f(&x);
    let mut func_evals = 1;
    let mut jac_evals = 0;
    let mut linear_solves = 0;

    // Calculate initial error
    let mut error = calculate_error(&residual, &params);

    // Check if already converged
    if error <= params.abs_tolerance {
        return Ok(NewtonResult {
            solution: x,
            residual,
            iterations: 0,
            converged: true,
            error_estimate: error,
            func_evals,
            jac_evals,
            linear_solves,
        });
    }

    // Create dummy timepoint - for ODE Jacobians we typically need time,
    // but for generic Newton solve we can use a dummy value
    let dummy_t = F::zero();

    // Iteration loop
    for iter in 0..params.max_iterations {
        // Update Jacobian if needed
        if iter == 0 && params.force_jacobian_init {
            // Force Jacobian update on first iteration if requested
            jac_manager.update_jacobian(dummy_t, &x, &|_t, y| f(&y.to_owned()), None)?;
            jac_evals += 1;
        } else if iter > 0 && iter % params.jacobian_update_freq == 0 {
            // Update Jacobian on specified frequency
            jac_manager.update_jacobian(dummy_t, &x, &|_t, y| f(&y.to_owned()), None)?;
            jac_evals += 1;
        } else if jac_manager.jacobian().is_none() {
            // Always update if no Jacobian exists
            jac_manager.update_jacobian(dummy_t, &x, &|_t, y| f(&y.to_owned()), None)?;
            jac_evals += 1;
        }

        // Get the Jacobian
        let jacobian = jac_manager.jacobian().unwrap();

        // Solve the linear system J * dx = -F(x) with optimized solver
        let neg_residual = residual.clone() * F::from_f64(-1.0).unwrap();

        // Use auto solver selection for the best performance
        let dx = auto_solve_linear_system(
            &jacobian.view(),
            &neg_residual.view(),
            LinearSolverType::Direct,
        )?;
        linear_solves += 1;

        // Apply damping if needed
        let mut damping = params.damping_factor;
        let mut x_new = x.clone() + &dx * damping;
        let mut residual_new = f(&x_new);
        func_evals += 1;

        // Calculate new error
        let mut error_new = calculate_error(&residual_new, &params);

        // Try to reduce step size if error increased
        // Simple backtracking line search
        if error_new > error && damping > params.min_damping {
            let mut backtrack_count = 0;

            while error_new > error && damping > params.min_damping && backtrack_count < 5 {
                damping *= F::from_f64(0.5).unwrap();
                x_new = x.clone() + &dx * damping;
                residual_new = f(&x_new);
                func_evals += 1;
                error_new = calculate_error(&residual_new, &params);
                backtrack_count += 1;
            }
        }

        // Update current point
        x = x_new;
        residual = residual_new;
        error = error_new;

        // Check convergence
        if error <= params.abs_tolerance || (error <= params.rel_tolerance * calculate_norm(&x)) {
            return Ok(NewtonResult {
                solution: x,
                residual,
                iterations: iter + 1,
                converged: true,
                error_estimate: error,
                func_evals,
                jac_evals,
                linear_solves,
            });
        }
    }

    // Failed to converge
    Err(IntegrateError::ConvergenceError(format!(
        "Newton's method failed to converge in {} iterations (error = {:.2e})",
        params.max_iterations, error,
    )))
}

/// Modified Newton solve that reuses the same Jacobian for multiple iterations
pub fn modified_newton_solve<F, Func>(
    f: Func,
    x0: Array1<F>,
    jac_manager: &mut JacobianManager<F>,
    params: NewtonParameters<F>,
) -> IntegrateResult<NewtonResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(&Array1<F>) -> Array1<F>,
{
    // Set large Jacobian update frequency
    let modified_params = NewtonParameters {
        jacobian_update_freq: params.max_iterations + 1, // Only update at beginning
        ..params
    };

    // Use standard Newton with modified parameters
    newton_solve(f, x0, jac_manager, modified_params)
}

/// Calculate error norm of residual vector
fn calculate_error<F: IntegrateFloat>(residual: &Array1<F>, _params: &NewtonParameters<F>) -> F {
    // Use L-infinity norm (max absolute value)
    let mut max_abs = F::zero();
    for &r in residual.iter() {
        max_abs = max_abs.max(r.abs());
    }
    max_abs
}

/// Calculate norm of a vector (for relative convergence)
fn calculate_norm<F: IntegrateFloat>(x: &Array1<F>) -> F {
    // Use L-infinity norm
    let mut max_abs = F::zero();
    for &val in x.iter() {
        max_abs = max_abs.max(val.abs());
    }
    max_abs
}
