//! SLSQP (Sequential Least SQuares Programming) algorithm for constrained optimization

use crate::constrained::{Constraint, ConstraintFn, ConstraintKind, Options};
use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1};

/// Implements the SLSQP algorithm for constrained optimization
#[allow(clippy::many_single_char_names)]
#[allow(dead_code)]
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

    // Separate constraints by type
    let mut ineq_constraints = Vec::new();
    let mut eq_constraints = Vec::new();
    for (i, constraint) in constraints.iter().enumerate() {
        if !constraint.is_bounds() {
            match constraint.kind {
                ConstraintKind::Inequality => ineq_constraints.push((i, constraint)),
                ConstraintKind::Equality => eq_constraints.push((i, constraint)),
            }
        }
    }

    let n_ineq = ineq_constraints.len();
    let n_eq = eq_constraints.len();

    // Initialize the Lagrange multipliers
    let mut lambda_ineq = Array1::zeros(n_ineq);
    let mut lambda_eq = Array1::zeros(n_eq);

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
    let mut c_ineq = Array1::zeros(n_ineq);
    let mut c_eq = Array1::zeros(n_eq);

    // Evaluate inequality constraints
    for (idx, (_, constraint)) in ineq_constraints.iter().enumerate() {
        let val = (constraint.fun)(x.as_slice().unwrap());
        c_ineq[idx] = val; // g(x) >= 0 constraint
        nfev += 1;
    }

    // Evaluate equality constraints
    for (idx, (_, constraint)) in eq_constraints.iter().enumerate() {
        let val = (constraint.fun)(x.as_slice().unwrap());
        c_eq[idx] = val; // h(x) = 0 constraint
        nfev += 1;
    }

    // Calculate constraint Jacobians separately
    let mut a_ineq = Array2::zeros((n_ineq, n));
    let mut a_eq = Array2::zeros((n_eq, n));

    // Inequality constraint Jacobian
    for (idx, (_, constraint)) in ineq_constraints.iter().enumerate() {
        for j in 0..n {
            let mut x_h = x.clone();
            x_h[j] += eps;
            let c_h = (constraint.fun)(x_h.as_slice().unwrap());
            a_ineq[[idx, j]] = (c_h - c_ineq[idx]) / eps;
            nfev += 1;
        }
    }

    // Equality constraint Jacobian
    for (idx, (_, constraint)) in eq_constraints.iter().enumerate() {
        for j in 0..n {
            let mut x_h = x.clone();
            x_h[j] += eps;
            let c_h = (constraint.fun)(x_h.as_slice().unwrap());
            a_eq[[idx, j]] = (c_h - c_eq[idx]) / eps;
            nfev += 1;
        }
    }

    // Initialize working matrices
    let mut h_inv = Array2::eye(n); // Approximate inverse Hessian

    // Main optimization loop
    let mut iter = 0;

    while iter < maxiter {
        // Check constraint violations
        let mut max_ineq_violation = 0.0;
        let mut max_eq_violation = 0.0;

        // Check inequality constraint violations
        for &ci in c_ineq.iter() {
            if ci < -ctol {
                max_ineq_violation = f64::max(max_ineq_violation, -ci);
            }
        }

        // Check equality constraint violations
        for &hi in c_eq.iter() {
            max_eq_violation = f64::max(max_eq_violation, hi.abs());
        }

        let max_constraint_violation = f64::max(max_ineq_violation, max_eq_violation);

        // Check convergence on gradient and constraints
        if g.iter().all(|&gi| gi.abs() < gtol) && max_constraint_violation < ctol {
            break;
        }

        // Compute the search direction using QP subproblem
        // For simplicity, we'll use a projected gradient approach
        let mut p = Array1::zeros(n);

        if max_constraint_violation > ctol {
            // If constraints are violated, move toward feasibility

            // Handle violated inequality constraints
            for (idx, &ci) in c_ineq.iter().enumerate() {
                if ci < -ctol {
                    // This inequality constraint is violated
                    for j in 0..n {
                        p[j] -= a_ineq[[idx, j]] * ci; // Move along constraint gradient
                    }
                }
            }

            // Handle violated equality constraints
            for (idx, &hi) in c_eq.iter().enumerate() {
                if hi.abs() > ctol {
                    // This equality constraint is violated
                    for j in 0..n {
                        p[j] -= a_eq[[idx, j]] * hi; // Move to satisfy h(x) = 0
                    }
                }
            }
        } else {
            // Otherwise, use BFGS direction with constraints
            p = -&h_inv.dot(&g);

            // Project gradient on active inequality constraints
            for (idx, &ci) in c_ineq.iter().enumerate() {
                if ci.abs() < ctol {
                    // Active inequality constraint
                    let mut normal = Array1::zeros(n);
                    for j in 0..n {
                        normal[j] = a_ineq[[idx, j]];
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

            // Project gradient on equality constraints (always active)
            for (idx, _) in c_eq.iter().enumerate() {
                let mut normal = Array1::zeros(n);
                for j in 0..n {
                    normal[j] = a_eq[[idx, j]];
                }
                let norm = normal.dot(&normal).sqrt();
                if norm > 1e-10 {
                    normal = &normal / norm;
                    let p_dot_normal = p.dot(&normal);

                    // Project out the component along equality constraint normal
                    p = &p - &(&normal * p_dot_normal);
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
        let mut c_ineq_new = Array1::zeros(n_ineq);
        let mut c_eq_new = Array1::zeros(n_eq);

        // Evaluate inequality constraints at new point
        for (idx, (_, constraint)) in ineq_constraints.iter().enumerate() {
            c_ineq_new[idx] = (constraint.fun)(x_new.as_slice().unwrap());
            nfev += 1;
        }

        // Evaluate equality constraints at new point
        for (idx, (_, constraint)) in eq_constraints.iter().enumerate() {
            c_eq_new[idx] = (constraint.fun)(x_new.as_slice().unwrap());
            nfev += 1;
        }

        // Check if constraint violation is reduced and objective decreases
        let max_viol = f64::max(max_ineq_violation, max_eq_violation);
        let mut max_viol_new = 0.0;

        // Check new inequality violations
        for &ci in c_ineq_new.iter() {
            max_viol_new = f64::max(max_viol_new, f64::max(0.0, -ci));
        }

        // Check new equality violations
        for &hi in c_eq_new.iter() {
            max_viol_new = f64::max(max_viol_new, hi.abs());
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
            for (idx, (_, constraint)) in ineq_constraints.iter().enumerate() {
                c_ineq_new[idx] = (constraint.fun)(x_new.as_slice().unwrap());
                nfev += 1;
            }

            for (idx, (_, constraint)) in eq_constraints.iter().enumerate() {
                c_eq_new[idx] = (constraint.fun)(x_new.as_slice().unwrap());
                nfev += 1;
            }

            max_viol_new = 0.0;
            for &ci in c_ineq_new.iter() {
                max_viol_new = f64::max(max_viol_new, f64::max(0.0, -ci));
            }
            for &hi in c_eq_new.iter() {
                max_viol_new = f64::max(max_viol_new, hi.abs());
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

        // Calculate new constraint Jacobians
        let mut a_ineq_new = Array2::zeros((n_ineq, n));
        let mut a_eq_new = Array2::zeros((n_eq, n));

        // New inequality constraint Jacobian
        for (idx, (_, constraint)) in ineq_constraints.iter().enumerate() {
            for j in 0..n {
                let mut x_h = x_new.clone();
                x_h[j] += eps;
                let c_h = (constraint.fun)(x_h.as_slice().unwrap());
                a_ineq_new[[idx, j]] = (c_h - c_ineq_new[idx]) / eps;
                nfev += 1;
            }
        }

        // New equality constraint Jacobian
        for (idx, (_, constraint)) in eq_constraints.iter().enumerate() {
            for j in 0..n {
                let mut x_h = x_new.clone();
                x_h[j] += eps;
                let c_h = (constraint.fun)(x_h.as_slice().unwrap());
                a_eq_new[[idx, j]] = (c_h - c_eq_new[idx]) / eps;
                nfev += 1;
            }
        }

        // Update Lagrange multipliers

        // Update inequality constraint multipliers
        for (idx, &ci) in c_ineq_new.iter().enumerate() {
            if ci.abs() < ctol {
                // For active inequality constraints
                let mut normal = Array1::zeros(n);
                for j in 0..n {
                    normal[j] = a_ineq_new[[idx, j]];
                }

                let norm = normal.dot(&normal).sqrt();
                if norm > 1e-10 {
                    normal = &normal / norm;
                    lambda_ineq[idx] = -g_new.dot(&normal);
                    // Ensure non-negativity for inequality multipliers
                    lambda_ineq[idx] = lambda_ineq[idx].max(0.0);
                }
            } else {
                lambda_ineq[idx] = 0.0;
            }
        }

        // Update equality constraint multipliers (can be any sign)
        for (idx, _) in c_eq_new.iter().enumerate() {
            let mut normal = Array1::zeros(n);
            for j in 0..n {
                normal[j] = a_eq_new[[idx, j]];
            }

            let norm = normal.dot(&normal).sqrt();
            if norm > 1e-10 {
                normal = &normal / norm;
                lambda_eq[idx] = -g_new.dot(&normal);
            }
        }

        // BFGS update for the Hessian approximation
        let s = &x_new - &x;
        let y = &g_new - &g;

        // Include constraints in y by adding Lagrangian term
        let mut y_lag = y.clone();

        // Add inequality constraint terms
        for (idx, &li) in lambda_ineq.iter().enumerate() {
            if li > 0.0 {
                // Only active inequality constraints
                for j in 0..n {
                    // L = f - sum(lambda_i * c_i)
                    // So gradient of L includes -lambda_i * grad(c_i)
                    y_lag[j] += li * (a_ineq_new[[idx, j]] - a_ineq[[idx, j]]);
                }
            }
        }

        // Add equality constraint terms (always active)
        for (idx, &li) in lambda_eq.iter().enumerate() {
            for j in 0..n {
                // L = f - sum(lambda_i * h_i)
                y_lag[j] += li * (a_eq_new[[idx, j]] - a_eq[[idx, j]]);
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
        c_ineq = c_ineq_new;
        c_eq = c_eq_new;
        a_ineq = a_ineq_new;
        a_eq = a_eq_new;

        iter += 1;
    }

    // Prepare constraint values for the result
    let mut c_result = Array1::zeros(constraints.len());

    // Fill inequality constraint values
    let mut ineq_idx = 0;
    let mut eq_idx = 0;
    for (i, constraint) in constraints.iter().enumerate() {
        if !constraint.is_bounds() {
            match constraint.kind {
                ConstraintKind::Inequality => {
                    c_result[i] = c_ineq[ineq_idx];
                    ineq_idx += 1;
                }
                ConstraintKind::Equality => {
                    c_result[i] = c_eq[eq_idx];
                    eq_idx += 1;
                }
            }
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
