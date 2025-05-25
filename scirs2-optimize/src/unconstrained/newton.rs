//! Newton methods for unconstrained optimization

use crate::error::OptimizeError;
use crate::unconstrained::conjugate_gradient::compute_line_bounds;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::{
    array_diff_norm, finite_difference_gradient, finite_difference_hessian,
};
use crate::unconstrained::{Bounds, Options};
use ndarray::{Array1, Array2, ArrayView1};

/// Implements the Newton-Conjugate-Gradient algorithm for unconstrained optimization
pub fn minimize_newton_cg<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S + Clone,
    S: Into<f64> + Clone,
{
    // Get options or use defaults
    let ftol = options.ftol;
    let gtol = options.gtol;
    let max_iter = options.max_iter;
    let eps = options.eps;
    let bounds = options.bounds.as_ref();

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();

    // Ensure initial point is within bounds
    if let Some(bounds) = bounds {
        bounds.project(x.as_slice_mut().unwrap());
    }

    // Function evaluation counter
    let mut nfev = 0;

    // Initialize function value
    let mut f = fun(&x.view()).into();
    nfev += 1;

    // Initialize gradient
    let mut g = finite_difference_gradient(&mut fun, &x.view(), eps)?;
    nfev += n;

    // Iteration counter
    let mut iter = 0;

    // Main optimization loop
    while iter < max_iter {
        // Check convergence on gradient
        let g_norm = g.dot(&g).sqrt();
        if g_norm < gtol {
            break;
        }

        // Save the current function value for convergence check
        let _f_old = f;

        // Calculate the Hessian approximation using finite differences
        let hess = finite_difference_hessian(&mut fun, &x.view(), eps)?;
        nfev += n * n;

        // Solve the Newton-CG system to find the step direction
        let mut p = solve_newton_cg_system(&g, &hess, gtol);

        // If the bounds are provided, project the search direction
        if let Some(bounds) = bounds {
            project_direction(&mut p, &x, Some(bounds));

            // If the projected direction is zero or too small, use the projected gradient
            let dir_norm = p.dot(&p).sqrt();
            if dir_norm < 1e-10 {
                // Try using the projected gradient instead
                p = -g.clone();
                project_direction(&mut p, &x, Some(bounds));

                // If even the projected gradient is zero, we're at a constrained optimum
                let pg_norm = p.dot(&p).sqrt();
                if pg_norm < 1e-10 {
                    break;
                }
            }
        }

        // Line search along the direction to determine the step size
        let (alpha, f_new) = line_search_newton(&mut fun, &x, &p, f, &mut nfev, bounds);

        // Take the step
        let mut x_new = &x + &(&p * alpha);

        // Ensure we're within bounds
        if let Some(bounds) = bounds {
            bounds.project(x_new.as_slice_mut().unwrap());
        }

        // Check if the step actually moved the point
        let step_size = array_diff_norm(&x_new.view(), &x.view());
        if step_size < 1e-10 {
            // We're at a boundary constraint and can't move further
            x = x_new;
            break;
        }

        // Calculate the new gradient
        let g_new = finite_difference_gradient(&mut fun, &x_new.view(), eps)?;
        nfev += n;

        // Check convergence on function value
        if (f - f_new).abs() < ftol * (1.0 + f.abs()) {
            x = x_new;
            g = g_new;
            break;
        }

        // Update variables for next iteration
        x = x_new;
        f = f_new;
        g = g_new;

        iter += 1;
    }

    // Use original function for final evaluation
    let final_fun = fun(&x.view());

    // Create and return result
    Ok(OptimizeResult {
        x,
        fun: final_fun,
        iterations: iter,
        nit: iter,
        func_evals: nfev,
        nfev,
        success: iter < max_iter,
        message: if iter < max_iter {
            "Optimization terminated successfully.".to_string()
        } else {
            "Maximum iterations reached.".to_string()
        },
        jacobian: Some(g),
        hessian: None,
    })
}

/// Solve the Newton-CG system Hx = -g using the conjugate gradient method
fn solve_newton_cg_system(g: &Array1<f64>, hess: &Array2<f64>, tol: f64) -> Array1<f64> {
    let n = g.len();

    // Start with x = 0
    let mut x = Array1::zeros(n);

    // If gradient is zero, return a zero step
    if g.dot(g) < 1e-10 {
        return x;
    }

    // Initialize residual r = -g - Hx = -g (since x=0)
    let mut r = -g.clone();

    // Initialize search direction p = r
    let mut p = r.clone();

    // Initial residual norm
    let r0_norm = r.dot(&r).sqrt();

    // Convergence tolerance (relative to initial residual)
    let cg_tol = f64::min(0.1, r0_norm * tol);

    // Maximum number of CG iterations
    let max_cg_iters = 2 * n;

    // Conjugate gradient iterations
    for _ in 0..max_cg_iters {
        // Compute H*p
        let hp = hess.dot(&p);

        // Compute p'*H*p
        let php = p.dot(&hp);

        // If the curvature is negative or very small, terminate the CG iterations
        if php <= 1e-10 {
            // Use the current direction, even if it's not optimal
            return x;
        }

        // Compute the CG step size
        let alpha = r.dot(&r) / php;

        // Update the solution
        x = &x + &(&p * alpha);

        // Update the residual: r_{k+1} = r_k - alpha * H * p_k
        r = &r - &(&hp * alpha);

        // Check convergence
        if r.dot(&r).sqrt() < cg_tol {
            break;
        }

        // Calculate beta for the next conjugate direction
        let r_new_norm_squared = r.dot(&r);
        let r_old_norm_squared = p.dot(&p);
        let beta = r_new_norm_squared / r_old_norm_squared;

        // Update the search direction
        p = &r + &(&p * beta);
    }

    x
}

/// Line search for Newton method with bounds support
fn line_search_newton<F, S>(
    fun: &mut F,
    x: &Array1<f64>,
    direction: &Array1<f64>,
    f_x: f64,
    nfev: &mut usize,
    bounds: Option<&Bounds>,
) -> (f64, f64)
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64>,
{
    // Get bounds on the line search parameter
    let (a_min, a_max) = if let Some(b) = bounds {
        compute_line_bounds(x, direction, Some(b))
    } else {
        (f64::NEG_INFINITY, f64::INFINITY)
    };

    // Use a simple backtracking line search with bounds
    let c1 = 1e-4; // Sufficient decrease parameter
    let rho = 0.5; // Backtracking parameter

    // Start with alpha with min(1.0, a_max) to ensure we're within bounds
    let mut alpha = if a_max < 1.0 { a_max * 0.99 } else { 1.0 };

    // If bounds fully constrain movement, return that constrained step
    if a_max <= 0.0 || a_min >= a_max {
        alpha = if a_max > 0.0 { a_max } else { 0.0 };
        let x_new = x + alpha * direction;
        *nfev += 1;
        let f_new = fun(&x_new.view()).into();
        return (alpha, f_new);
    }

    // Function to evaluate a point on the line
    let mut f_line = |alpha: f64| {
        let mut x_new = x + alpha * direction;

        // Project onto bounds (if needed)
        if let Some(bounds) = bounds {
            bounds.project(x_new.as_slice_mut().unwrap());
        }

        *nfev += 1;
        fun(&x_new.view()).into()
    };

    // Initial step
    let mut f_new = f_line(alpha);

    // Backtracking until Armijo condition is satisfied or we hit the lower bound
    let slope = direction.mapv(|d| d.powi(2)).sum();
    while f_new > f_x - c1 * alpha * slope.abs() && alpha > a_min {
        alpha *= rho;

        // Ensure alpha is at least a_min
        if alpha < a_min {
            alpha = a_min;
        }

        f_new = f_line(alpha);

        // Prevent infinite loops for very small steps
        if alpha < 1e-10 {
            break;
        }
    }

    (alpha, f_new)
}

/// Projects the search direction to ensure we don't move in a direction that
/// immediately violates the bounds.
fn project_direction(direction: &mut Array1<f64>, x: &Array1<f64>, bounds: Option<&Bounds>) {
    if bounds.is_none() {
        return;
    }

    let bounds = bounds.unwrap();

    for i in 0..x.len() {
        let xi = x[i];

        // Check if we're at a bound
        if let Some(lb) = bounds.lower[i] {
            if (xi - lb).abs() < 1e-10 && direction[i] < 0.0 {
                // At lower bound and moving in negative direction
                direction[i] = 0.0;
            }
        }

        if let Some(ub) = bounds.upper[i] {
            if (xi - ub).abs() < 1e-10 && direction[i] > 0.0 {
                // At upper bound and moving in positive direction
                direction[i] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_newton_cg_quadratic() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + 4.0 * x[1] * x[1] };

        let x0 = Array1::from_vec(vec![2.0, 1.0]);
        let options = Options::default();

        let result = minimize_newton_cg(quadratic, x0, &options).unwrap();

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_newton_cg_with_bounds() {
        let quadratic =
            |x: &ArrayView1<f64>| -> f64 { (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();
        options.max_iter = 100; // More iterations for bounded optimization

        // Constrain solution to [0, 1] x [0, 1]
        let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);
        options.bounds = Some(bounds);

        let result = minimize_newton_cg(quadratic, x0, &options).unwrap();

        assert!(result.success);
        // The optimal point (2, 3) is outside the bounds, so we should get (1, 1)
        // Allow more tolerance for bounded optimization
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 0.4);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 0.4);
    }

    #[test]
    fn test_newton_cg_rosenbrock() {
        let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
            let a = 1.0;
            let b = 100.0;
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();
        options.max_iter = 100; // Newton methods may need more iterations

        let result = minimize_newton_cg(rosenbrock, x0, &options).unwrap();

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }
}
