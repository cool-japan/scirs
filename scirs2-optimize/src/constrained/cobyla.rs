//! COBYLA (Constrained Optimization BY Linear Approximations) algorithm

use crate::constrained::{Constraint, ConstraintFn, ConstraintKind, Options};
use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix1};
use scirs2_core::validation::check_finite;

/// COBYLA optimizer for constrained optimization
///
/// COBYLA is a numerical optimization method for constrained problems where
/// the objective function and the constraints are not required to be differentiable.
/// It works by building linear approximations of the objective function and
/// constraints, then solving linear programming subproblems.
///
/// # Arguments
///
/// * `func` - The objective function to minimize
/// * `x0` - Initial guess for the solution
/// * `constraints` - Vector of constraint functions
/// * `options` - Optimization options
///
/// # References
///
/// Powell, M. J. D. (1994). "A Direct Search Optimization Method That Models
/// the Objective and Constraint Functions by Linear Interpolation."
/// In Advances in Optimization and Numerical Analysis, pp. 51-67.
#[allow(dead_code)]
pub fn minimize_cobyla<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint<ConstraintFn>],
    options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
{
    let n = x0.len();
    let x0_vec = x0.to_vec();

    // Validate inputs
    for (i, &val) in x0_vec.iter().enumerate() {
        check_finite(val, format!("x0[{}]", i))?;
    }
    if n == 0 {
        return Err(OptimizeError::ValueError("x0 cannot be empty".to_string()));
    }

    // Set up algorithm parameters
    let maxiter = options.maxiter.unwrap_or(1000);
    let ftol = options.ftol.unwrap_or(1e-8);
    let ctol = options.ctol.unwrap_or(1e-8);

    // Initial trust region radius
    let mut rho = 0.1;
    let rho_end = 1e-8;
    let rho_beg = 0.5;

    if rho_beg < rho_end {
        return Err(OptimizeError::ValueError(
            "Initial trust region radius must be >= final radius".to_string(),
        ));
    }

    // Initialize state
    let mut x = Array1::from_vec(x0_vec);
    let mut f = func(&x.to_vec());
    let mut nfev = 1;
    let mut nit = 0;

    // Count constraints
    let num_constraints = constraints.len();

    // Evaluate initial constraints
    let mut constraint_values = Array1::zeros(num_constraints);
    for (i, constraint) in constraints.iter().enumerate() {
        constraint_values[i] = (constraint.fun)(&x.to_vec());
        nfev += 1;
    }

    // Check initial constraint satisfaction
    let mut max_constraint_violation: f64 = 0.0;
    for (i, constraint) in constraints.iter().enumerate() {
        let violation = match constraint.kind {
            ConstraintKind::Equality => constraint_values[i].abs(),
            ConstraintKind::Inequality => {
                if constraint_values[i] < 0.0 {
                    -constraint_values[i]
                } else {
                    0.0
                }
            }
        };
        max_constraint_violation = max_constraint_violation.max(violation);
    }

    // Build initial interpolation set
    let npt = 2 * n + 1; // Number of interpolation points
    let mut xpt = Array2::zeros((npt, n)); // Interpolation points
    let mut fval = Array1::zeros(npt); // Function values at interpolation points
    let mut con = Array2::zeros((npt, num_constraints)); // Constraint values

    // First point is the starting point
    xpt.row_mut(0).assign(&x);
    fval[0] = f;
    con.row_mut(0).assign(&constraint_values);

    // Create additional interpolation points
    for i in 1..=n {
        let mut xi = x.clone();
        xi[i - 1] += rho;
        xpt.row_mut(i).assign(&xi);
        fval[i] = func(&xi.to_vec());
        nfev += 1;

        for (j, constraint) in constraints.iter().enumerate() {
            con[[i, j]] = (constraint.fun)(&xi.to_vec());
            nfev += 1;
        }
    }

    for i in (n + 1)..npt {
        let mut xi = x.clone();
        xi[i - n - 1] -= rho;
        xpt.row_mut(i).assign(&xi);
        fval[i] = func(&xi.to_vec());
        nfev += 1;

        for (j, constraint) in constraints.iter().enumerate() {
            con[[i, j]] = (constraint.fun)(&xi.to_vec());
            nfev += 1;
        }
    }

    // Main optimization loop
    let mut success = false;
    let mut message = "Maximum number of iterations reached".to_string();

    while nit < maxiter && rho > rho_end {
        nit += 1;

        // Build linear models for objective and constraints
        let (grad_f, grad_c) = build_linear_models(&xpt, &fval, &con, &x, n, num_constraints)?;

        // Solve trust region subproblem
        let step = solve_trust_region_subproblem(&grad_f, &grad_c, constraints, rho, n)?;

        // Compute trial point
        let mut x_trial = x.clone();
        for i in 0..n {
            x_trial[i] += step[i];
        }

        // Evaluate objective and constraints at trial point
        let f_trial = func(&x_trial.to_vec());
        nfev += 1;

        let mut c_trial = Array1::zeros(num_constraints);
        for (i, constraint) in constraints.iter().enumerate() {
            c_trial[i] = (constraint.fun)(&x_trial.to_vec());
            nfev += 1;
        }

        // Compute predicted reduction
        let pred_reduction = -grad_f.dot(&step);
        let actual_reduction = f - f_trial;

        // Compute constraint violation
        let mut trial_violation: f64 = 0.0;
        for (i, constraint) in constraints.iter().enumerate() {
            let violation = match constraint.kind {
                ConstraintKind::Equality => c_trial[i].abs(),
                ConstraintKind::Inequality => {
                    if c_trial[i] < 0.0 {
                        -c_trial[i]
                    } else {
                        0.0
                    }
                }
            };
            trial_violation = trial_violation.max(violation);
        }

        // Accept or reject the step
        let ratio = if pred_reduction.abs() > 1e-15 {
            actual_reduction / pred_reduction
        } else {
            0.0
        };

        let accept_step = ratio > 0.1 && trial_violation <= max_constraint_violation * 1.1;

        if accept_step {
            x = x_trial;
            f = f_trial;
            constraint_values = c_trial;
            max_constraint_violation = trial_violation;

            // Update interpolation set
            update_interpolation_set(&mut xpt, &mut fval, &mut con, &x, f, &constraint_values, 0);
        }

        // Update trust region radius
        if ratio > 0.75 && step.dot(&step).sqrt() > 0.9 * rho {
            rho = (2.0 * rho).min(rho_beg);
        } else if ratio <= 0.25 {
            rho *= 0.5;
        }

        // Check convergence
        if max_constraint_violation <= ctol && step.dot(&step).sqrt() <= ftol {
            success = true;
            message = "Optimization terminated successfully".to_string();
            break;
        }

        if rho <= rho_end {
            if max_constraint_violation <= ctol {
                success = true;
                message = "Optimization terminated successfully".to_string();
            } else {
                message = "Trust region radius became too small".to_string();
            }
            break;
        }
    }

    Ok(OptimizeResults::<f64> {
        x,
        fun: f,
        jac: None,
        hess: None,
        constr: Some(constraint_values),
        nit,
        nfev,
        njev: 0,
        nhev: 0,
        maxcv: 0,
        message,
        success,
        status: if success { 0 } else { 1 },
    })
}

/// Build linear models for the objective function and constraints
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn build_linear_models(
    xpt: &Array2<f64>,
    fval: &Array1<f64>,
    con: &Array2<f64>,
    _x: &Array1<f64>,
    n: usize,
    num_constraints: usize,
) -> OptimizeResult<(Array1<f64>, Array2<f64>)> {
    // Simple finite difference approximation for gradients
    let mut grad_f = Array1::zeros(n);
    let mut grad_c = Array2::zeros((num_constraints, n));

    let _h = 1e-8;

    // Build gradient of objective function
    for i in 0..n {
        if i + 1 < xpt.nrows() {
            grad_f[i] = (fval[i + 1] - fval[0]) / (xpt[[i + 1, i]] - xpt[[0, i]]);
        }
    }

    // Build gradients of constraint functions
    for j in 0..num_constraints {
        for i in 0..n {
            if i + 1 < xpt.nrows() {
                grad_c[[j, i]] = (con[[i + 1, j]] - con[[0, j]]) / (xpt[[i + 1, i]] - xpt[[0, i]]);
            }
        }
    }

    Ok((grad_f, grad_c))
}

/// Solve the trust region subproblem
#[allow(dead_code)]
fn solve_trust_region_subproblem(
    grad_f: &Array1<f64>,
    grad_c: &Array2<f64>,
    constraints: &[Constraint<ConstraintFn>],
    rho: f64,
    n: usize,
) -> OptimizeResult<Array1<f64>> {
    // Simple approach: project the negative gradient onto the feasible region
    let mut step = -grad_f.clone();

    // Scale to trust region
    let step_norm = step.dot(&step).sqrt();
    if step_norm > rho {
        step *= rho / step_norm;
    }

    // Simple constraint handling: if step violates linear constraints, scale it back
    for (i, constraint) in constraints.iter().enumerate() {
        if i < grad_c.nrows() {
            let grad_ci = grad_c.row(i);
            let pred_constraint = grad_ci.dot(&step);

            match constraint.kind {
                ConstraintKind::Inequality => {
                    if pred_constraint < -0.1 * rho {
                        // Scale back the step
                        let scale = -0.1 * rho / pred_constraint;
                        step *= scale;
                    }
                }
                ConstraintKind::Equality => {
                    // For equality constraints, project the step
                    let norm_sq = grad_ci.dot(&grad_ci);
                    if norm_sq > 1e-15 {
                        let projection = pred_constraint / norm_sq;
                        for j in 0..n {
                            step[j] -= projection * grad_ci[j];
                        }
                    }
                }
            }
        }
    }

    Ok(step)
}

/// Update the interpolation set with a new point
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn update_interpolation_set(
    xpt: &mut Array2<f64>,
    fval: &mut Array1<f64>,
    con: &mut Array2<f64>,
    x: &Array1<f64>,
    f: f64,
    constraint_values: &Array1<f64>,
    index: usize,
) {
    // Update the interpolation point
    xpt.row_mut(index).assign(x);
    fval[index] = f;
    con.row_mut(index).assign(constraint_values);
}

// Implement error conversion for validation errors
impl From<scirs2_core::error::CoreError> for OptimizeError {
    fn from(error: scirs2_core::error::CoreError) -> Self {
        OptimizeError::ValueError(error.to_string())
    }
}
