//! Trust-region algorithm for constrained optimization
//!
//! This module implements the trust-region constrained optimization algorithm,
//! supporting both numerical and user-supplied gradients and Hessians.
//!
//! # Example with custom gradient
//!
//! ```
//! use scirs2_optimize::constrained::trust_constr::{
//!     minimize_trust_constr_with_derivatives, HessianUpdate,
//! };
//! use scirs2_optimize::constrained::{Constraint, Options};
//! use scirs2_core::ndarray::{array, Array1};
//!
//! // Objective: f(x) = (x[0] - 1)² + (x[1] - 2.5)²
//! fn objective(x: &[f64]) -> f64 {
//!     (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2)
//! }
//!
//! // Analytical gradient: [2(x[0] - 1), 2(x[1] - 2.5)]
//! fn gradient(x: &[f64]) -> Array1<f64> {
//!     array![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.5)]
//! }
//!
//! // Constraint: x[0] + x[1] <= 3  =>  g(x) = 3 - x[0] - x[1] >= 0
//! fn constraint(x: &[f64]) -> f64 {
//!     3.0 - x[0] - x[1]
//! }
//!
//! let x0 = array![0.0, 0.0];
//! let constraints = vec![Constraint::new(constraint, Constraint::INEQUALITY)];
//! let options = Options::default();
//!
//! let result = minimize_trust_constr_with_derivatives(
//!     objective,
//!     &x0,
//!     &constraints,
//!     &options,
//!     Some(gradient),           // Custom gradient
//!     None::<fn(&[f64]) -> _>,  // No custom Hessian (use BFGS)
//!     HessianUpdate::BFGS,
//! ).expect("valid input");
//!
//! assert!(result.success);
//! ```

use crate::constrained::{Constraint, ConstraintKind, Options};
use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1};

/// Fill one row of the constraint Jacobian using the constraint's analytical
/// Jacobian when present, otherwise forward finite differences.
///
/// Returns the number of constraint evaluations performed (so the caller can
/// keep `nfev` accurate). On an analytical-Jacobian length mismatch, falls back
/// to finite differences (no panic, no unwrap).
fn fill_constraint_jac_row<S>(
    a: &mut Array2<f64>,
    row: usize,
    constraint: &Constraint,
    x: &ArrayBase<S, Ix1>,
    c_val: f64,
    n: usize,
    eps: f64,
) -> usize
where
    S: Data<Elem = f64>,
{
    if let Some(ref jac_fn) = constraint.jac {
        let grad = jac_fn(x.as_slice().expect("Operation failed"));
        if grad.len() == n {
            for j in 0..n {
                a[[row, j]] = grad[j];
            }
            return 0;
        }
        // Length mismatch: fall through to finite differences.
    }

    let mut evals = 0;
    for j in 0..n {
        let mut x_h = x.to_owned();
        x_h[j] += eps;
        let c_h = (constraint.fun)(x_h.as_slice().expect("Operation failed"));
        a[[row, j]] = (c_h - c_val) / eps;
        evals += 1;
    }
    evals
}

/// Hessian update strategy for quasi-Newton methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HessianUpdate {
    /// BFGS update (default) - good for general problems
    #[default]
    BFGS,
    /// SR1 update - can capture negative curvature
    SR1,
    /// Use exact Hessian provided by user
    Exact,
}

/// Type alias for gradient function
pub type GradientFn = fn(&[f64]) -> Array1<f64>;

/// Type alias for Hessian function
pub type HessianFn = fn(&[f64]) -> Array2<f64>;

/// Minimizes a function with constraints using the trust-region method.
///
/// `obj_jac` optionally supplies an analytical objective gradient (issue #127);
/// when `None`, finite differences are used. Per-constraint analytical
/// Jacobians attached via [`Constraint::with_jacobian`] are honoured.
#[allow(clippy::many_single_char_names)]
#[allow(dead_code)]
pub fn minimize_trust_constr<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint],
    obj_jac: Option<&dyn Fn(&[f64]) -> Array1<f64>>,
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
{
    // Get options or use defaults
    let ftol = options.ftol.unwrap_or(1e-8);
    let gtol = options.gtol.unwrap_or(1e-8);
    let ctol = options.ctol.unwrap_or(1e-8);
    let maxiter = options.maxiter.unwrap_or(100 * x0.len());
    let eps = options.eps.unwrap_or(1e-8);

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();
    let mut f = func(x.as_slice().expect("Operation failed"));
    let mut nfev = 1;

    // Initialize the Lagrange multipliers
    let mut lambda = Array1::zeros(constraints.len());

    // Calculate initial gradient (analytical if supplied, else finite differences)
    let mut g = if let Some(grad_fn) = obj_jac {
        grad_fn(x.as_slice().expect("Operation failed"))
    } else {
        let mut grad = Array1::zeros(n);
        for i in 0..n {
            let mut x_h = x.clone();
            x_h[i] += eps;
            let f_h = func(x_h.as_slice().expect("Operation failed"));
            grad[i] = (f_h - f) / eps;
            nfev += 1;
        }
        grad
    };

    // Evaluate initial constraints
    let mut c = Array1::zeros(constraints.len());
    for (i, constraint) in constraints.iter().enumerate() {
        if !constraint.is_bounds() {
            let val = (constraint.fun)(x.as_slice().expect("Operation failed"));

            match constraint.kind {
                ConstraintKind::Inequality => {
                    c[i] = val; // g(x) >= 0 constraint
                }
                ConstraintKind::Equality => {
                    c[i] = val; // h(x) = 0 constraint
                }
            }
        }
    }

    // Calculate constraint Jacobian (analytical per-constraint if attached)
    let mut a = Array2::zeros((constraints.len(), n));
    for (i, constraint) in constraints.iter().enumerate() {
        if !constraint.is_bounds() {
            nfev += fill_constraint_jac_row(&mut a, i, constraint, &x, c[i], n, eps);
        }
    }

    // Initialize trust region radius
    let mut delta = 1.0;

    // Initialize approximation of the Hessian of the Lagrangian
    let mut b = Array2::eye(n);

    // Main optimization loop
    let mut iter = 0;

    while iter < maxiter {
        // Check constraint violation
        let mut max_constraint_violation = 0.0;
        for (i, &ci) in c.iter().enumerate() {
            match constraints[i].kind {
                ConstraintKind::Inequality => {
                    if ci < -ctol {
                        max_constraint_violation = f64::max(max_constraint_violation, -ci);
                    }
                }
                ConstraintKind::Equality => {
                    // For equality constraints, violation is |h(x)|
                    let violation = ci.abs();
                    if violation > ctol {
                        max_constraint_violation = f64::max(max_constraint_violation, violation);
                    }
                }
            }
        }

        // Check convergence on gradient and constraints
        if g.iter().all(|&gi| gi.abs() < gtol) && max_constraint_violation < ctol {
            break;
        }

        // Compute the Lagrangian gradient
        let mut lag_grad = g.clone();
        for (i, &li) in lambda.iter().enumerate() {
            let include_constraint = match constraints[i].kind {
                ConstraintKind::Inequality => {
                    // For inequality constraints: include if active (lambda > 0) or violated (c < 0)
                    li > 0.0 || c[i] < -ctol
                }
                ConstraintKind::Equality => {
                    // For equality constraints: always include (always active)
                    true
                }
            };

            if include_constraint {
                for j in 0..n {
                    // L = f - sum(lambda_i * c_i) for constraints
                    // So gradient of L is grad(f) - sum(lambda_i * grad(c_i))
                    lag_grad[j] -= li * a[[i, j]];
                }
            }
        }

        // Compute the step using a trust-region approach
        // Solve the constrained quadratic subproblem:
        // min 0.5 * p^T B p + g^T p  s.t. ||p|| <= delta and linearized constraints

        let (p, predicted_reduction) =
            compute_trust_region_step_constrained(&lag_grad, &b, &a, &c, delta, constraints, ctol);

        // Try the step
        let x_new = &x + &p;

        // Evaluate function and constraints at new point
        let f_new = func(x_new.as_slice().expect("Operation failed"));
        nfev += 1;

        let mut c_new = Array1::zeros(constraints.len());
        for (i, constraint) in constraints.iter().enumerate() {
            if !constraint.is_bounds() {
                c_new[i] = (constraint.fun)(x_new.as_slice().expect("Operation failed"));
                nfev += 1;
            }
        }

        // Compute change in merit function (includes constraint violation)
        let mut merit = f;
        let mut merit_new = f_new;

        // Add constraint violation penalty
        let penalty = 10.0; // Simple fixed penalty parameter
        for (i, &ci) in c.iter().enumerate() {
            match constraints[i].kind {
                ConstraintKind::Inequality => {
                    // For inequality constraints: penalize only violations (g(x) < 0)
                    merit += penalty * f64::max(0.0, -ci);
                    merit_new += penalty * f64::max(0.0, -c_new[i]);
                }
                ConstraintKind::Equality => {
                    // For equality constraints: penalize any deviation from zero
                    merit += penalty * ci.abs();
                    merit_new += penalty * c_new[i].abs();
                }
            }
        }

        // Compute actual reduction in merit function
        let actual_reduction = merit - merit_new;

        // Compute ratio of actual to predicted reduction
        let rho = if predicted_reduction > 0.0 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };

        // Update trust region radius based on the quality of the step
        if rho < 0.25 {
            delta *= 0.5;
        } else if rho > 0.75 && p.iter().map(|&pi| pi * pi).sum::<f64>().sqrt() >= 0.9 * delta {
            delta *= 2.0;
        }

        // Accept or reject the step
        if rho > 0.1 {
            // Accept the step
            x = x_new;
            f = f_new;
            c = c_new;

            // Check convergence on function value
            if (merit - merit_new).abs() < ftol * (1.0 + merit.abs()) {
                break;
            }

            // Compute new gradient (analytical if supplied, else finite differences)
            let g_new = if let Some(grad_fn) = obj_jac {
                grad_fn(x.as_slice().expect("Operation failed"))
            } else {
                let mut grad = Array1::zeros(n);
                for i in 0..n {
                    let mut x_h = x.clone();
                    x_h[i] += eps;
                    let f_h = func(x_h.as_slice().expect("Operation failed"));
                    grad[i] = (f_h - f) / eps;
                    nfev += 1;
                }
                grad
            };

            // Compute new constraint Jacobian (analytical per-constraint if attached)
            let mut a_new = Array2::zeros((constraints.len(), n));
            for (i, constraint) in constraints.iter().enumerate() {
                if !constraint.is_bounds() {
                    nfev += fill_constraint_jac_row(&mut a_new, i, constraint, &x, c[i], n, eps);
                }
            }

            // Update Lagrange multipliers using projected gradient method
            for (i, constraint) in constraints.iter().enumerate() {
                match constraint.kind {
                    ConstraintKind::Inequality => {
                        if c[i] < -ctol {
                            // Active or violated constraint
                            // Increase multiplier if constraint is violated
                            lambda[i] = f64::max(0.0, lambda[i] - c[i] * penalty);
                        } else {
                            // Decrease multiplier towards zero
                            lambda[i] = f64::max(0.0, lambda[i] - 0.1 * lambda[i]);
                        }
                    }
                    ConstraintKind::Equality => {
                        // For equality constraints, multipliers can be positive or negative
                        // Update based on constraint violation to drive h(x) -> 0
                        let step_size = 0.1;
                        lambda[i] -= step_size * c[i] * penalty;

                        // Optional: add some damping to prevent oscillations
                        lambda[i] *= 0.9;
                    }
                }
            }

            // Update Hessian approximation using BFGS or SR1
            let s = &p;
            let y = &g_new - &g;

            // Simple BFGS update for the Hessian approximation
            let s_dot_y = s.dot(&y);
            if s_dot_y > 1e-10 {
                let s_col = s.clone().insert_axis(Axis(1));
                let s_row = s.clone().insert_axis(Axis(0));

                let bs = b.dot(s);
                let bs_col = bs.clone().insert_axis(Axis(1));
                let bs_row = bs.clone().insert_axis(Axis(0));

                let term1 = s_dot_y + s.dot(&bs);
                let term2 = &s_col.dot(&s_row) * (term1 / (s_dot_y * s_dot_y));

                let term3 = &bs_col.dot(&s_row) + &s_col.dot(&bs_row);

                b = &b + &term2 - &(&term3 / s_dot_y);
            }

            // Update variables for next iteration
            g = g_new;
            a = a_new;
        }

        iter += 1;
    }

    // Prepare constraint values for the result
    let mut c_result = Array1::zeros(constraints.len());
    for (i, constraint) in constraints.iter().enumerate() {
        if !constraint.is_bounds() {
            c_result[i] = c[i];
        }
    }

    // Create and return result
    let mut result = OptimizeResults::default();
    result.x = x;
    result.fun = f;
    result.jac = Some(g.into_raw_vec_and_offset().0);
    result.constr = Some(c_result);
    result.nfev = nfev;
    result.nit = iter;
    result.success = iter < maxiter;

    if result.success {
        result.message = "Optimization terminated successfully.".to_string();
    } else {
        result.message = "Maximum iterations reached.".to_string();
    }

    Ok(result)
}

/// Minimizes a function with constraints using trust-region method with custom derivatives.
///
/// This function extends [`minimize_trust_constr`] by allowing user-supplied gradient
/// and Hessian functions, which can significantly improve performance and accuracy
/// compared to finite-difference approximations.
///
/// # Arguments
///
/// * `func` - The objective function to minimize
/// * `x0` - Initial guess for the optimization
/// * `constraints` - Vector of constraints (equality and/or inequality)
/// * `options` - Optimization options (tolerances, max iterations, etc.)
/// * `jac` - Optional gradient function. If `None`, finite differences are used.
/// * `hess` - Optional Hessian function. Only used when `hess_update` is [`HessianUpdate::Exact`].
/// * `hess_update` - Strategy for Hessian updates (BFGS, SR1, or Exact)
///
/// # Returns
///
/// * `OptimizeResults` containing the optimization solution
///
/// # Example
///
/// ```
/// use scirs2_optimize::constrained::trust_constr::{
///     minimize_trust_constr_with_derivatives, HessianUpdate,
/// };
/// use scirs2_optimize::constrained::{Constraint, Options};
/// use scirs2_core::ndarray::{array, Array1, Array2};
///
/// // Rosenbrock function
/// fn rosenbrock(x: &[f64]) -> f64 {
///     (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
/// }
///
/// // Analytical gradient
/// fn rosenbrock_grad(x: &[f64]) -> Array1<f64> {
///     array![
///         -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
///         200.0 * (x[1] - x[0].powi(2))
///     ]
/// }
///
/// // Constraint: x[0]^2 + x[1]^2 <= 2
/// fn circle_constraint(x: &[f64]) -> f64 {
///     2.0 - x[0].powi(2) - x[1].powi(2)
/// }
///
/// let x0 = array![0.0, 0.0];
/// let constraints = vec![Constraint::new(circle_constraint, Constraint::INEQUALITY)];
///
/// let result = minimize_trust_constr_with_derivatives(
///     rosenbrock,
///     &x0,
///     &constraints,
///     &Options::default(),
///     Some(rosenbrock_grad),
///     None::<fn(&[f64]) -> Array2<f64>>,
///     HessianUpdate::BFGS,
/// ).expect("valid input");
/// ```
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn minimize_trust_constr_with_derivatives<F, S, G, H>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint],
    options: &Options,
    jac: Option<G>,
    hess: Option<H>,
    hess_update: HessianUpdate,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
    G: Fn(&[f64]) -> Array1<f64>,
    H: Fn(&[f64]) -> Array2<f64>,
{
    // Get options or use defaults
    let ftol = options.ftol.unwrap_or(1e-8);
    let gtol = options.gtol.unwrap_or(1e-8);
    let ctol = options.ctol.unwrap_or(1e-8);
    let maxiter = options.maxiter.unwrap_or(100 * x0.len());
    let eps = options.eps.unwrap_or(1e-8);

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();
    let mut f = func(x.as_slice().expect("Operation failed"));
    let mut nfev = 1;
    let mut njev = 0;
    let mut nhev = 0;

    // Initialize the Lagrange multipliers
    let mut lambda = Array1::zeros(constraints.len());

    // Calculate initial gradient (using custom jac if provided)
    let mut g = if let Some(ref grad_fn) = jac {
        njev += 1;
        grad_fn(x.as_slice().expect("Operation failed"))
    } else {
        // Finite difference gradient
        let mut grad = Array1::zeros(n);
        for i in 0..n {
            let mut x_h = x.clone();
            x_h[i] += eps;
            let f_h = func(x_h.as_slice().expect("Operation failed"));
            grad[i] = (f_h - f) / eps;
            nfev += 1;
        }
        grad
    };

    // Evaluate initial constraints
    let mut c = Array1::zeros(constraints.len());
    for (i, constraint) in constraints.iter().enumerate() {
        if !constraint.is_bounds() {
            let val = (constraint.fun)(x.as_slice().expect("Operation failed"));
            match constraint.kind {
                ConstraintKind::Inequality => c[i] = val,
                ConstraintKind::Equality => c[i] = val,
            }
        }
    }

    // Calculate constraint Jacobian (analytical per-constraint if attached)
    let mut a = Array2::zeros((constraints.len(), n));
    for (i, constraint) in constraints.iter().enumerate() {
        if !constraint.is_bounds() {
            nfev += fill_constraint_jac_row(&mut a, i, constraint, &x, c[i], n, eps);
        }
    }

    // Initialize trust region radius
    let mut delta = 1.0;

    // Initialize Hessian or its approximation
    let mut b = if hess_update == HessianUpdate::Exact {
        if let Some(ref hess_fn) = hess {
            nhev += 1;
            hess_fn(x.as_slice().expect("Operation failed"))
        } else {
            // Fallback to identity if exact requested but no Hessian provided
            Array2::eye(n)
        }
    } else {
        Array2::eye(n)
    };

    // Main optimization loop
    let mut iter = 0;

    while iter < maxiter {
        // Check constraint violation
        let mut max_constraint_violation = 0.0;
        for (i, &ci) in c.iter().enumerate() {
            match constraints[i].kind {
                ConstraintKind::Inequality => {
                    if ci < -ctol {
                        max_constraint_violation = f64::max(max_constraint_violation, -ci);
                    }
                }
                ConstraintKind::Equality => {
                    let violation = ci.abs();
                    if violation > ctol {
                        max_constraint_violation = f64::max(max_constraint_violation, violation);
                    }
                }
            }
        }

        // Check convergence
        if g.iter().all(|&gi| gi.abs() < gtol) && max_constraint_violation < ctol {
            break;
        }

        // Compute the Lagrangian gradient
        let mut lag_grad = g.clone();
        for (i, &li) in lambda.iter().enumerate() {
            let include_constraint = match constraints[i].kind {
                ConstraintKind::Inequality => li > 0.0 || c[i] < -ctol,
                ConstraintKind::Equality => true,
            };

            if include_constraint {
                for j in 0..n {
                    lag_grad[j] -= li * a[[i, j]];
                }
            }
        }

        // Compute the step
        let (p, predicted_reduction) =
            compute_trust_region_step_constrained(&lag_grad, &b, &a, &c, delta, constraints, ctol);

        // Try the step
        let x_new = &x + &p;
        let f_new = func(x_new.as_slice().expect("Operation failed"));
        nfev += 1;

        let mut c_new = Array1::zeros(constraints.len());
        for (i, constraint) in constraints.iter().enumerate() {
            if !constraint.is_bounds() {
                c_new[i] = (constraint.fun)(x_new.as_slice().expect("Operation failed"));
                nfev += 1;
            }
        }

        // Compute merit function
        let mut merit = f;
        let mut merit_new = f_new;
        let penalty = 10.0;

        for (i, &ci) in c.iter().enumerate() {
            match constraints[i].kind {
                ConstraintKind::Inequality => {
                    merit += penalty * f64::max(0.0, -ci);
                    merit_new += penalty * f64::max(0.0, -c_new[i]);
                }
                ConstraintKind::Equality => {
                    merit += penalty * ci.abs();
                    merit_new += penalty * c_new[i].abs();
                }
            }
        }

        let actual_reduction = merit - merit_new;
        let rho = if predicted_reduction > 0.0 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };

        // Update trust region radius
        if rho < 0.25 {
            delta *= 0.5;
        } else if rho > 0.75 && p.iter().map(|&pi| pi * pi).sum::<f64>().sqrt() >= 0.9 * delta {
            delta *= 2.0;
        }

        // Accept or reject the step
        if rho > 0.1 {
            x = x_new;
            f = f_new;
            c = c_new;

            if (merit - merit_new).abs() < ftol * (1.0 + merit.abs()) {
                break;
            }

            // Compute new gradient
            let g_new = if let Some(ref grad_fn) = jac {
                njev += 1;
                grad_fn(x.as_slice().expect("Operation failed"))
            } else {
                let mut grad = Array1::zeros(n);
                for i in 0..n {
                    let mut x_h = x.clone();
                    x_h[i] += eps;
                    let f_h = func(x_h.as_slice().expect("Operation failed"));
                    grad[i] = (f_h - f) / eps;
                    nfev += 1;
                }
                grad
            };

            // Compute new constraint Jacobian (analytical per-constraint if attached)
            let mut a_new = Array2::zeros((constraints.len(), n));
            for (i, constraint) in constraints.iter().enumerate() {
                if !constraint.is_bounds() {
                    nfev += fill_constraint_jac_row(&mut a_new, i, constraint, &x, c[i], n, eps);
                }
            }

            // Update Lagrange multipliers
            for (i, constraint) in constraints.iter().enumerate() {
                match constraint.kind {
                    ConstraintKind::Inequality => {
                        if c[i] < -ctol {
                            lambda[i] = f64::max(0.0, lambda[i] - c[i] * penalty);
                        } else {
                            lambda[i] = f64::max(0.0, lambda[i] - 0.1 * lambda[i]);
                        }
                    }
                    ConstraintKind::Equality => {
                        let step_size = 0.1;
                        lambda[i] -= step_size * c[i] * penalty;
                        lambda[i] *= 0.9;
                    }
                }
            }

            // Update Hessian based on strategy
            match hess_update {
                HessianUpdate::Exact => {
                    if let Some(ref hess_fn) = hess {
                        nhev += 1;
                        b = hess_fn(x.as_slice().expect("Operation failed"));
                    }
                }
                HessianUpdate::BFGS => {
                    let s = &p;
                    let y = &g_new - &g;
                    let s_dot_y = s.dot(&y);
                    if s_dot_y > 1e-10 {
                        let s_col = s.clone().insert_axis(Axis(1));
                        let s_row = s.clone().insert_axis(Axis(0));
                        let bs = b.dot(s);
                        let bs_col = bs.clone().insert_axis(Axis(1));
                        let bs_row = bs.clone().insert_axis(Axis(0));
                        let term1 = s_dot_y + s.dot(&bs);
                        let term2 = &s_col.dot(&s_row) * (term1 / (s_dot_y * s_dot_y));
                        let term3 = &bs_col.dot(&s_row) + &s_col.dot(&bs_row);
                        b = &b + &term2 - &(&term3 / s_dot_y);
                    }
                }
                HessianUpdate::SR1 => {
                    let s = &p;
                    let y = &g_new - &g;
                    let bs = b.dot(s);
                    let diff = &y - &bs;
                    let s_dot_diff = s.dot(&diff);
                    // SR1 update with skipping condition
                    if s_dot_diff.abs() > 1e-8 * s.dot(s).sqrt() * diff.dot(&diff).sqrt() {
                        let diff_col = diff.clone().insert_axis(Axis(1));
                        let diff_row = diff.clone().insert_axis(Axis(0));
                        let update = &diff_col.dot(&diff_row) / s_dot_diff;
                        b = &b + &update;
                    }
                }
            }

            g = g_new;
            a = a_new;
        }

        iter += 1;
    }

    // Prepare result
    let mut c_result = Array1::zeros(constraints.len());
    for (i, constraint) in constraints.iter().enumerate() {
        if !constraint.is_bounds() {
            c_result[i] = c[i];
        }
    }

    let mut result = OptimizeResults::default();
    result.x = x;
    result.fun = f;
    result.jac = Some(g.into_raw_vec_and_offset().0);
    result.constr = Some(c_result);
    result.nfev = nfev;
    result.njev = njev;
    result.nhev = nhev;
    result.nit = iter;
    result.success = iter < maxiter;

    if result.success {
        result.message = "Optimization terminated successfully.".to_string();
    } else {
        result.message = "Maximum iterations reached.".to_string();
    }

    Ok(result)
}

/// Compute a trust-region step for constrained optimization
#[allow(clippy::many_single_char_names)]
#[allow(dead_code)]
fn compute_trust_region_step_constrained(
    g: &Array1<f64>,
    b: &Array2<f64>,
    a: &Array2<f64>,
    c: &Array1<f64>,
    delta: f64,
    constraints: &[Constraint],
    ctol: f64,
) -> (Array1<f64>, f64) {
    let n = g.len();
    let n_constr = constraints.len();

    // Compute the unconstrained Cauchy point (steepest descent direction)
    let p_unconstrained = compute_unconstrained_cauchy_point(g, b, delta);

    // Check if unconstrained Cauchy point satisfies linearized constraints
    let mut constraint_violated = false;
    for i in 0..n_constr {
        let grad_c_dot_p = (0..n).map(|j| a[[i, j]] * p_unconstrained[j]).sum::<f64>();

        match constraints[i].kind {
            ConstraintKind::Inequality => {
                // For inequality constraints: check if g(x) + grad_g^T p >= -ctol
                if c[i] + grad_c_dot_p < -ctol {
                    constraint_violated = true;
                    break;
                }
            }
            ConstraintKind::Equality => {
                // For equality constraints: check if |h(x) + grad_h^T p| <= ctol
                if (c[i] + grad_c_dot_p).abs() > ctol {
                    constraint_violated = true;
                    break;
                }
            }
        }
    }

    // If unconstrained point is feasible, use it
    if !constraint_violated {
        // Compute predicted reduction
        let g_dot_p = g.dot(&p_unconstrained);
        let bp = b.dot(&p_unconstrained);
        let p_dot_bp = p_unconstrained.dot(&bp);
        let predicted_reduction = -g_dot_p - 0.5 * p_dot_bp;

        return (p_unconstrained, predicted_reduction);
    }

    // Otherwise, project onto the linearized feasible region
    // This is a simplified approach - in practice, you would solve a CQP

    // Start with the steepest descent direction
    let mut p = Array1::zeros(n);
    for i in 0..n {
        p[i] = -g[i];
    }

    // Normalize to trust region radius
    let p_norm = p.iter().map(|&pi| pi * pi).sum::<f64>().sqrt();
    if p_norm > 1e-10 {
        p = &p * (delta / p_norm);
    }

    // Project onto each constraint
    for _iter in 0..5 {
        // Limited iterations for projection
        let mut max_viol = 0.0;
        let mut most_violated = 0;

        // Find most violated constraint
        for i in 0..n_constr {
            let grad_c_dot_p = (0..n).map(|j| a[[i, j]] * p[j]).sum::<f64>();

            let viol = match constraints[i].kind {
                ConstraintKind::Inequality => {
                    // For inequality constraints: violation is max(0, -(g + grad_g^T p))
                    f64::max(0.0, -(c[i] + grad_c_dot_p))
                }
                ConstraintKind::Equality => {
                    // For equality constraints: violation is |h + grad_h^T p|
                    (c[i] + grad_c_dot_p).abs()
                }
            };

            if viol > max_viol {
                max_viol = viol;
                most_violated = i;
            }
        }

        if max_viol < ctol {
            break;
        }

        // Project p onto the constraint
        let mut a_norm_sq = 0.0;
        for j in 0..n {
            a_norm_sq += a[[most_violated, j]] * a[[most_violated, j]];
        }

        if a_norm_sq > 1e-10 {
            let grad_c_dot_p = (0..n).map(|j| a[[most_violated, j]] * p[j]).sum::<f64>();

            let proj_dist = match constraints[most_violated].kind {
                ConstraintKind::Inequality => {
                    // For inequality constraints: project to boundary when violated
                    if c[most_violated] + grad_c_dot_p < 0.0 {
                        -(c[most_violated] + grad_c_dot_p) / a_norm_sq
                    } else {
                        0.0
                    }
                }
                ConstraintKind::Equality => {
                    // For equality constraints: always project to satisfy h + grad_h^T p = 0
                    -(c[most_violated] + grad_c_dot_p) / a_norm_sq
                }
            };

            // Project p
            for j in 0..n {
                p[j] += a[[most_violated, j]] * proj_dist;
            }

            // Rescale to trust region
            let p_norm = p.iter().map(|&pi| pi * pi).sum::<f64>().sqrt();
            if p_norm > delta {
                p = &p * (delta / p_norm);
            }
        }
    }

    // Compute predicted reduction
    let g_dot_p = g.dot(&p);
    let bp = b.dot(&p);
    let p_dot_bp = p.dot(&bp);
    let predicted_reduction = -g_dot_p - 0.5 * p_dot_bp;

    (p, predicted_reduction)
}

/// Compute the unconstrained Cauchy point (steepest descent to trust region boundary)
#[allow(dead_code)]
fn compute_unconstrained_cauchy_point(g: &Array1<f64>, b: &Array2<f64>, delta: f64) -> Array1<f64> {
    let n = g.len();

    // Compute the steepest descent direction: -g
    let mut p = Array1::zeros(n);
    for i in 0..n {
        p[i] = -g[i];
    }

    // Compute g^T B g and ||g||^2
    let bg = b.dot(g);
    let g_dot_bg = g.dot(&bg);
    let g_norm_sq = g.dot(g);

    // Check if the gradient is practically zero
    if g_norm_sq < 1e-10 {
        // If gradient is practically zero, don't move
        return Array1::zeros(n);
    }

    // Compute tau (step to the boundary)
    let tau = if g_dot_bg <= 0.0 {
        // Negative curvature or zero curvature case
        delta / g_norm_sq.sqrt()
    } else {
        // Positive curvature case
        f64::min(delta / g_norm_sq.sqrt(), g_norm_sq / g_dot_bg)
    };

    // Scale the direction by tau
    for i in 0..n {
        p[i] *= tau;
    }

    p
}
