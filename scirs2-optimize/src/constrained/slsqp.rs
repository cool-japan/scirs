//! SLSQP (Sequential Least SQuares Programming) algorithm for constrained optimization

use crate::constrained::{Constraint, ConstraintFn, ConstraintKind, Options};
use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1};

/// Implements the SLSQP algorithm for constrained optimization
pub fn minimize_slsqp<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint<ConstraintFn>],
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
    let mut f = func(x.as_slice().unwrap());
    let mut nfev = 1;

    // Initialize the Lagrange multipliers for inequality constraints
    let mut lambda = Array1::zeros(constraints.len());

    // Calculate initial gradient using finite differences
    let mut g = Array1::zeros(n);
    for i in 0..n {
        let mut x_h = x.clone();
        x_h[i] += eps;
        let f_h = func(x_h.as_slice().unwrap());
        g[i] = (f_h - f) / eps;
        nfev += 1;
    }

    // Evaluate initial constraints
    let mut c = Array1::zeros(constraints.len());
    let _ceq: Array1<f64> = Array1::zeros(0); // No equality constraints support for now
    for (i, constraint) in constraints.iter().enumerate() {
        if !constraint.is_bounds() {
            let val = (constraint.fun)(x.as_slice().unwrap());

            match constraint.kind {
                ConstraintKind::Inequality => {
                    c[i] = val; // g(x) >= 0 constraint
                }
                ConstraintKind::Equality => {
                    // For simplicity, we don't fully support equality constraints yet
                    return Err(OptimizeError::NotImplementedError(
                        "Equality constraints not fully implemented in SLSQP yet".to_string(),
                    ));
                }
            }
        }
    }

    // Calculate constraint Jacobian
    let mut a = Array2::zeros((constraints.len(), n));
    for (i, constraint) in constraints.iter().enumerate() {
        if !constraint.is_bounds() {
            for j in 0..n {
                let mut x_h = x.clone();
                x_h[j] += eps;
                let c_h = (constraint.fun)(x_h.as_slice().unwrap());
                a[[i, j]] = (c_h - c[i]) / eps;
                nfev += 1;
            }
        }
    }

    // Initialize working matrices
    let mut h_inv = Array2::eye(n); // Approximate inverse Hessian

    // Main optimization loop
    let mut iter = 0;

    while iter < maxiter {
        // Check constraint violation
        let mut max_constraint_violation = 0.0;
        for (i, &ci) in c.iter().enumerate() {
            if constraints[i].kind == ConstraintKind::Inequality && ci < -ctol {
                max_constraint_violation = f64::max(max_constraint_violation, -ci);
            }
        }

        // Check convergence on gradient
        if g.iter().all(|&gi| gi.abs() < gtol) && max_constraint_violation < ctol {
            break;
        }

        // Compute the search direction using QP subproblem
        // For simplicity, we'll use a projected gradient approach
        let mut p = Array1::zeros(n);

        if max_constraint_violation > ctol {
            // If constraints are violated, move toward feasibility
            for (i, &ci) in c.iter().enumerate() {
                if ci < -ctol {
                    // This constraint is violated
                    for j in 0..n {
                        p[j] -= a[[i, j]] * ci; // Move along constraint gradient
                    }
                }
            }
        } else {
            // Otherwise, use BFGS direction with constraints
            p = -&h_inv.dot(&g);

            // Project gradient on active constraints
            for (i, &ci) in c.iter().enumerate() {
                if ci.abs() < ctol {
                    // Active constraint
                    let mut normal = Array1::zeros(n);
                    for j in 0..n {
                        normal[j] = a[[i, j]];
                    }
                    let norm = normal.dot(&normal).sqrt();
                    if norm > 1e-10 {
                        normal = &normal / norm;
                        let p_dot_normal = p.dot(&normal);

                        // If moving in the wrong direction (constraint violation), project out
                        if p_dot_normal < 0.0 {
                            p = &p - &(&normal * p_dot_normal);
                        }
                    }
                }
            }
        }

        // Line search with constraint awareness
        let mut alpha = 1.0;
        let c1 = 1e-4; // Sufficient decrease parameter
        let rho = 0.5; // Backtracking parameter

        // Initial step
        let mut x_new = &x + &(&p * alpha);
        let mut f_new = func(x_new.as_slice().unwrap());
        nfev += 1;

        // Evaluate constraints at new point
        let mut c_new = Array1::zeros(constraints.len());
        for (i, constraint) in constraints.iter().enumerate() {
            if !constraint.is_bounds() {
                c_new[i] = (constraint.fun)(x_new.as_slice().unwrap());
                nfev += 1;
            }
        }

        // Check if constraint violation is reduced and objective decreases
        let mut max_viol = 0.0;
        let mut max_viol_new = 0.0;

        for (i, constraint) in constraints.iter().enumerate() {
            if constraint.kind == ConstraintKind::Inequality {
                max_viol = f64::max(max_viol, f64::max(0.0, -c[i]));
                max_viol_new = f64::max(max_viol_new, f64::max(0.0, -c_new[i]));
            }
        }

        // Compute directional derivative
        let g_dot_p = g.dot(&p);

        // Backtracking line search
        while (f_new > f + c1 * alpha * g_dot_p && max_viol <= ctol)
            || (max_viol_new > max_viol && max_viol > ctol)
        {
            alpha *= rho;

            // Prevent tiny steps
            if alpha < 1e-10 {
                break;
            }

            x_new = &x + &(&p * alpha);
            f_new = func(x_new.as_slice().unwrap());
            nfev += 1;

            // Evaluate constraints
            for (i, constraint) in constraints.iter().enumerate() {
                if !constraint.is_bounds() {
                    c_new[i] = (constraint.fun)(x_new.as_slice().unwrap());
                    nfev += 1;
                }
            }

            max_viol_new = 0.0;
            for (i, constraint) in constraints.iter().enumerate() {
                if constraint.kind == ConstraintKind::Inequality {
                    max_viol_new = f64::max(max_viol_new, f64::max(0.0, -c_new[i]));
                }
            }
        }

        // Check convergence on function value and step size
        if ((f - f_new).abs() < ftol * (1.0 + f.abs())) && alpha * p.dot(&p).sqrt() < ftol {
            break;
        }

        // Calculate new gradient
        let mut g_new = Array1::zeros(n);
        for i in 0..n {
            let mut x_h = x_new.clone();
            x_h[i] += eps;
            let f_h = func(x_h.as_slice().unwrap());
            g_new[i] = (f_h - f_new) / eps;
            nfev += 1;
        }

        // Calculate new constraint Jacobian
        let mut a_new = Array2::zeros((constraints.len(), n));
        for (i, constraint) in constraints.iter().enumerate() {
            if !constraint.is_bounds() {
                for j in 0..n {
                    let mut x_h = x_new.clone();
                    x_h[j] += eps;
                    let c_h = (constraint.fun)(x_h.as_slice().unwrap());
                    a_new[[i, j]] = (c_h - c_new[i]) / eps;
                    nfev += 1;
                }
            }
        }

        // Update Lagrange multipliers (rudimentary)
        for (i, constraint) in constraints.iter().enumerate() {
            if constraint.kind == ConstraintKind::Inequality && c_new[i].abs() < ctol {
                // For active inequality constraints
                let mut normal = Array1::zeros(n);
                for j in 0..n {
                    normal[j] = a_new[[i, j]];
                }

                let norm = normal.dot(&normal).sqrt();
                if norm > 1e-10 {
                    normal = &normal / norm;
                    lambda[i] = -g_new.dot(&normal);
                }
            } else {
                lambda[i] = 0.0;
            }
        }

        // BFGS update for the Hessian approximation
        let s = &x_new - &x;
        let y = &g_new - &g;

        // Include constraints in y by adding Lagrangian term
        let mut y_lag = y.clone();
        for (i, &li) in lambda.iter().enumerate() {
            if li > 0.0 {
                // Only active constraints
                for j in 0..n {
                    // L = f - sum(lambda_i * c_i)
                    // So gradient of L includes -lambda_i * grad(c_i)
                    y_lag[j] += li * (a_new[[i, j]] - a[[i, j]]);
                }
            }
        }

        // BFGS update formula
        let rho_bfgs = 1.0 / y_lag.dot(&s);
        if rho_bfgs.is_finite() && rho_bfgs > 0.0 {
            let i_mat = Array2::eye(n);
            let y_row = y_lag.clone().insert_axis(Axis(0));
            let s_col = s.clone().insert_axis(Axis(1));
            let y_s_t = y_row.dot(&s_col);

            let term1 = &i_mat - &(&y_s_t * rho_bfgs);
            let s_row = s.clone().insert_axis(Axis(0));
            let y_col = y_lag.clone().insert_axis(Axis(1));
            let s_y_t = s_row.dot(&y_col);

            let term2 = &i_mat - &(&s_y_t * rho_bfgs);
            let term3 = &term1.dot(&h_inv);
            h_inv = term3.dot(&term2) + rho_bfgs * s_col.dot(&s_row);
        }

        // Update for next iteration
        x = x_new;
        f = f_new;
        g = g_new;
        c = c_new;
        a = a_new;

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
