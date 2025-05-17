//! Trust-region algorithm for constrained optimization

use crate::constrained::{Constraint, ConstraintFn, ConstraintKind, Options};
use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1};

pub fn minimize_trust_constr<F, S>(
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

    // Initialize the Lagrange multipliers
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
                        "Equality constraints not fully implemented in Trust-Region yet"
                            .to_string(),
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
            if constraints[i].kind == ConstraintKind::Inequality && ci < -ctol {
                max_constraint_violation = f64::max(max_constraint_violation, -ci);
            }
        }

        // Check convergence on gradient and constraints
        if g.iter().all(|&gi| gi.abs() < gtol) && max_constraint_violation < ctol {
            break;
        }

        // Compute the Lagrangian gradient
        let mut lag_grad = g.clone();
        for (i, &li) in lambda.iter().enumerate() {
            if li > 0.0 || c[i] < ctol {
                // Active or violated constraint
                for j in 0..n {
                    // L = f - sum(lambda_i * c_i) for inequality constraints
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
        let f_new = func(x_new.as_slice().unwrap());
        nfev += 1;

        let mut c_new = Array1::zeros(constraints.len());
        for (i, constraint) in constraints.iter().enumerate() {
            if !constraint.is_bounds() {
                c_new[i] = (constraint.fun)(x_new.as_slice().unwrap());
                nfev += 1;
            }
        }

        // Compute change in merit function (includes constraint violation)
        let mut merit = f;
        let mut merit_new = f_new;

        // Add constraint violation penalty
        let penalty = 10.0; // Simple fixed penalty parameter
        for (i, &ci) in c.iter().enumerate() {
            if constraints[i].kind == ConstraintKind::Inequality {
                merit += penalty * f64::max(0.0, -ci);
                merit_new += penalty * f64::max(0.0, -c_new[i]);
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

            // Compute new gradient
            let mut g_new = Array1::zeros(n);
            for i in 0..n {
                let mut x_h = x.clone();
                x_h[i] += eps;
                let f_h = func(x_h.as_slice().unwrap());
                g_new[i] = (f_h - f) / eps;
                nfev += 1;
            }

            // Compute new constraint Jacobian
            let mut a_new = Array2::zeros((constraints.len(), n));
            for (i, constraint) in constraints.iter().enumerate() {
                if !constraint.is_bounds() {
                    for j in 0..n {
                        let mut x_h = x.clone();
                        x_h[j] += eps;
                        let c_h = (constraint.fun)(x_h.as_slice().unwrap());
                        a_new[[i, j]] = (c_h - c[i]) / eps;
                        nfev += 1;
                    }
                }
            }

            // Update Lagrange multipliers using projected gradient method
            for (i, constraint) in constraints.iter().enumerate() {
                if constraint.kind == ConstraintKind::Inequality {
                    if c[i] < ctol {
                        // Active or violated constraint
                        // Increase multiplier if constraint is violated
                        lambda[i] = f64::max(0.0, lambda[i] - c[i] * penalty);
                    } else {
                        // Decrease multiplier towards zero
                        lambda[i] = f64::max(0.0, lambda[i] - 0.1 * lambda[i]);
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

/// Compute a trust-region step for constrained optimization
fn compute_trust_region_step_constrained(
    g: &Array1<f64>,
    b: &Array2<f64>,
    a: &Array2<f64>,
    c: &Array1<f64>,
    delta: f64,
    constraints: &[Constraint<ConstraintFn>],
    ctol: f64,
) -> (Array1<f64>, f64) {
    let n = g.len();
    let n_constr = constraints.len();

    // Compute the unconstrained Cauchy point (steepest descent direction)
    let p_unconstrained = compute_unconstrained_cauchy_point(g, b, delta);

    // Check if unconstrained Cauchy point satisfies linearized constraints
    let mut constraint_violated = false;
    for i in 0..n_constr {
        if constraints[i].kind == ConstraintKind::Inequality {
            let grad_c_dot_p = (0..n).map(|j| a[[i, j]] * p_unconstrained[j]).sum::<f64>();
            if c[i] + grad_c_dot_p < -ctol {
                constraint_violated = true;
                break;
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
            if constraints[i].kind == ConstraintKind::Inequality {
                let grad_c_dot_p = (0..n).map(|j| a[[i, j]] * p[j]).sum::<f64>();
                let viol = -(c[i] + grad_c_dot_p);
                if viol > max_viol {
                    max_viol = viol;
                    most_violated = i;
                }
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
            let proj_dist = (c[most_violated] + grad_c_dot_p) / a_norm_sq;

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
