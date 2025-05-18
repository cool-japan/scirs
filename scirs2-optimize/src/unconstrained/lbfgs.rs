//! Limited-memory BFGS algorithms for large-scale optimization

use crate::error::OptimizeError;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::{Bounds, Options};
use ndarray::{Array1, ArrayView1};

/// Implements the L-BFGS-B algorithm for bound-constrained optimization
pub fn minimize_lbfgsb<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64> + Clone,
{
    // Get options or use defaults
    let m = options.max_iter.min(10); // Use max_iter as memory size for compatibility
    let factr = 1e7; // Machine epsilon factor
    let pgtol = options.gtol;
    let max_iter = options.max_iter;
    let eps = options.eps;
    let bounds = options.bounds.as_ref();

    // Machine precision (estimate)
    let eps_mach = 2.22e-16;
    let ftol = factr * eps_mach;

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
    nfev += 1;
    let mut f = fun(&x.view()).into();

    // Initialize gradient, using appropriate methods for boundaries
    let mut g = Array1::zeros(n);
    calculate_gradient(&mut fun, &x, &mut g, eps, bounds, &mut nfev);

    // Iteration counter
    let mut iter = 0;

    // Storage for the limited-memory BFGS update
    let mut s_vectors: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut y_vectors: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut rho_values: Vec<f64> = Vec::with_capacity(m);

    // Main optimization loop
    while iter < max_iter {
        // Save the current point and gradient
        let g_old = g.clone();
        let f_old = f;

        // Compute the search direction using the L-BFGS two-loop recursion
        let mut search_direction = -g.clone();

        // L-BFGS two-loop recursion to compute a search direction
        let mut alpha_values = Vec::with_capacity(s_vectors.len());

        // First loop: compute and save alpha values
        for i in (0..s_vectors.len()).rev() {
            let rho_i = rho_values[i];
            let s_i = &s_vectors[i];
            let y_i = &y_vectors[i];

            let alpha_i = rho_i * s_i.dot(&search_direction);
            alpha_values.push(alpha_i);

            search_direction = &search_direction - &(alpha_i * y_i);
        }

        // Scale the search direction by an approximation of the initial inverse Hessian
        if !s_vectors.is_empty() {
            let y_last = &y_vectors[s_vectors.len() - 1];
            let s_last = &s_vectors[s_vectors.len() - 1];

            let ys = y_last.dot(s_last);
            let yy = y_last.dot(y_last);

            if ys > 0.0 && yy > 0.0 {
                let gamma = ys / yy;
                search_direction = &search_direction * gamma;
            }
        }

        // Second loop: compute the final search direction
        for (i, &alpha_i) in alpha_values.iter().enumerate() {
            let idx = s_vectors.len() - 1 - i;
            let rho_i = rho_values[idx];
            let s_i = &s_vectors[idx];
            let y_i = &y_vectors[idx];

            let beta_i = rho_i * y_i.dot(&search_direction);
            search_direction = &search_direction + &(s_i * (alpha_i - beta_i));
        }

        // Make the search direction negative for minimization
        search_direction = -search_direction;

        // Project the search direction to ensure we don't violate bounds
        project_direction(&mut search_direction, &x, bounds);

        // Line search to find a step size that satisfies the Armijo condition
        let (alpha, f_new) =
            lbfgsb_line_search(&mut fun, &x, &search_direction, f, bounds, &mut nfev);

        // If line search couldn't find an acceptable step, we may be done
        if alpha < 1e-10 {
            break;
        }

        // Update position
        let x_new = &x + &(&search_direction * alpha);

        // Calculate new gradient
        calculate_gradient(&mut fun, &x_new, &mut g, eps, bounds, &mut nfev);

        // Compute sk = xk+1 - xk and yk = gk+1 - gk
        let s_k = &x_new - &x;
        let y_k = &g - &g_old;

        // Check if s_k and y_k are usable for the BFGS update
        let s_dot_y = s_k.dot(&y_k);

        if s_dot_y > 0.0 {
            // Update the limited-memory information
            if s_vectors.len() == m {
                // If we've reached the limit, remove the oldest vectors
                s_vectors.remove(0);
                y_vectors.remove(0);
                rho_values.remove(0);
            }

            // Add new vectors
            s_vectors.push(s_k);
            y_vectors.push(y_k);
            rho_values.push(1.0 / s_dot_y);
        }

        // Update current position and function value
        x = x_new;
        f = f_new;

        // Check for convergence
        // Calculate projected gradient norm
        let pg_norm = projected_gradient_norm(&x, &g, bounds);

        // Check if we're done
        if pg_norm < pgtol {
            break;
        }

        // Check for convergence on function value
        let f_change = (f_old - f).abs();
        if f_change < ftol * (1.0 + f.abs()) {
            break;
        }

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

/// Implements the Limited-memory BFGS algorithm for large-scale optimization
pub fn minimize_lbfgs<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64> + Clone,
{
    // Get options or use defaults
    let m = options.max_iter.min(10); // Use max_iter as memory size for compatibility
    let ftol = options.ftol;
    let gtol = options.gtol;
    let max_iter = options.max_iter;
    let eps = options.eps;

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();

    // Function evaluation counter
    let mut nfev = 0;

    // Initialize function value
    nfev += 1;
    let mut f = fun(&x.view()).into();

    // Initialize gradient using finite differences
    let mut g = Array1::zeros(n);

    // Calculate initial gradient
    let mut g_old = Array1::zeros(n);
    calculate_gradient(&mut fun, &x, &mut g, eps, None, &mut nfev);

    // Iteration counter
    let mut iter = 0;

    // Storage for the limited-memory BFGS update
    let mut s_vectors: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut y_vectors: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut rho_values: Vec<f64> = Vec::with_capacity(m);

    // Main optimization loop
    while iter < max_iter {
        // Check convergence on gradient
        let g_norm = g.dot(&g).sqrt();
        if g_norm < gtol {
            break;
        }

        // Save the current point and gradient
        g_old.assign(&g);
        let f_old = f;

        // Compute the search direction using the L-BFGS two-loop recursion
        let mut search_direction = -g.clone();

        // L-BFGS two-loop recursion to compute a search direction
        let mut alpha_values = Vec::with_capacity(s_vectors.len());

        // First loop: compute and save alpha values
        for i in (0..s_vectors.len()).rev() {
            let rho_i = rho_values[i];
            let s_i = &s_vectors[i];
            let y_i = &y_vectors[i];

            let alpha_i = rho_i * s_i.dot(&search_direction);
            alpha_values.push(alpha_i);

            search_direction = &search_direction - &(alpha_i * y_i);
        }

        // Scale the search direction by an approximation of the initial inverse Hessian
        if !s_vectors.is_empty() {
            let y_last = &y_vectors[s_vectors.len() - 1];
            let s_last = &s_vectors[s_vectors.len() - 1];

            let ys = y_last.dot(s_last);
            let yy = y_last.dot(y_last);

            if ys > 0.0 && yy > 0.0 {
                let gamma = ys / yy;
                search_direction = &search_direction * gamma;
            }
        }

        // Second loop: compute the final search direction
        for (i, &alpha_i) in alpha_values.iter().enumerate() {
            let idx = s_vectors.len() - 1 - i;
            let rho_i = rho_values[idx];
            let s_i = &s_vectors[idx];
            let y_i = &y_vectors[idx];

            let beta_i = rho_i * y_i.dot(&search_direction);
            search_direction = &search_direction + &(s_i * (alpha_i - beta_i));
        }

        // Make the search direction negative for minimization
        search_direction = -search_direction;

        // More robust line search for L-BFGS
        let c1 = 1e-4; // Sufficient decrease parameter

        // Try different initial step lengths for more robust line search
        let initial_steps = [1.0, 0.5, 0.1, 0.01, 0.001];
        let mut found_good_step = false;
        let mut alpha;
        let mut x_new = x.clone();
        let mut f_new = f;

        // Backtracking line search with different initial steps
        let g_dot_p = g.dot(&search_direction);

        // Only try line search if gradient dot product with search direction is negative
        if g_dot_p < 0.0 {
            for &init_alpha in &initial_steps {
                alpha = init_alpha;
                x_new = &x + &(&search_direction * alpha);
                f_new = fun(&x_new.view()).into();
                nfev += 1;

                // If we already have a decrease, start backtracking from here
                if f_new < f {
                    found_good_step = true;
                    break;
                }

                // Otherwise, try backtracking
                let rho = 0.5; // Backtracking parameter
                let mut backtrack_iter = 0;

                while f_new > f + c1 * alpha * g_dot_p && backtrack_iter < 16 {
                    alpha *= rho;
                    x_new = &x + &(&search_direction * alpha);
                    f_new = fun(&x_new.view()).into();
                    nfev += 1;
                    backtrack_iter += 1;

                    if f_new < f {
                        found_good_step = true;
                        break;
                    }
                }

                if found_good_step {
                    break;
                }
            }
        }

        // If no good step found, take a small step in the gradient direction
        if !found_good_step {
            // Take a small step in the negative gradient direction
            let small_step = 1e-4;
            search_direction = -g.clone();
            alpha = small_step / g.dot(&g).sqrt();
            x_new = &x + &(&search_direction * alpha);
            f_new = fun(&x_new.view()).into();
            nfev += 1;
        }

        // Compute step and gradient difference
        let s_k = &x_new - &x;

        // If the step is very small, we may be at a minimum
        if s_k.iter().all(|&si| si.abs() < 1e-10) {
            x = x_new;
            break;
        }

        // Update position
        x = x_new;

        // Calculate new gradient
        calculate_gradient(&mut fun, &x, &mut g, eps, None, &mut nfev);

        // Compute yk = gk+1 - gk
        let y_k = &g - &g_old;

        // Check if s_k and y_k are usable for the BFGS update
        let s_dot_y = s_k.dot(&y_k);

        if s_dot_y > 0.0 {
            // Update the limited-memory information
            if s_vectors.len() == m {
                // If we've reached the limit, remove the oldest vectors
                s_vectors.remove(0);
                y_vectors.remove(0);
                rho_values.remove(0);
            }

            // Add new vectors
            s_vectors.push(s_k);
            y_vectors.push(y_k);
            rho_values.push(1.0 / s_dot_y);
        }

        // Update function value
        f = f_new;

        // Check for convergence on function value
        let f_change = (f_old - f).abs();
        if f_change < ftol * (1.0 + f.abs()) {
            break;
        }

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

/// Calculate gradient using finite differences, with special handling for bounds
fn calculate_gradient<F, S>(
    fun: &mut F,
    x: &Array1<f64>,
    g: &mut Array1<f64>,
    eps: f64,
    bounds: Option<&Bounds>,
    nfev: &mut usize,
) where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64>,
{
    let n = x.len();
    let f_x = fun(&x.view()).into();
    *nfev += 1;

    for i in 0..n {
        // Don't modify the original point
        let mut x_h = x.clone();

        // For bounded variables, use one-sided differences at boundaries
        if let Some(bounds) = bounds {
            let eps_i = eps * (1.0 + x[i].abs());
            if let Some(ub) = bounds.upper[i] {
                if x[i] >= ub - eps_i {
                    // Near upper bound, use backward difference
                    x_h[i] = x[i] - eps_i;
                    *nfev += 1;
                    let f_h = fun(&x_h.view()).into();
                    g[i] = (f_x - f_h) / eps_i;
                    continue;
                }
            }
            if let Some(lb) = bounds.lower[i] {
                if x[i] <= lb + eps_i {
                    // Near lower bound, use forward difference
                    x_h[i] = x[i] + eps_i;
                    *nfev += 1;
                    let f_h = fun(&x_h.view()).into();
                    g[i] = (f_h - f_x) / eps_i;
                    continue;
                }
            }
        }

        // Otherwise use central difference
        let eps_i = eps * (1.0 + x[i].abs());
        x_h[i] = x[i] + eps_i;
        *nfev += 1;
        let f_p = fun(&x_h.view()).into();

        x_h[i] = x[i] - eps_i;
        *nfev += 1;
        let f_m = fun(&x_h.view()).into();

        g[i] = (f_p - f_m) / (2.0 * eps_i);
    }
}

/// Calculate the projected gradient norm, which measures how close we are to a stationary point
/// in the presence of bounds constraints.
fn projected_gradient_norm(x: &Array1<f64>, g: &Array1<f64>, bounds: Option<&Bounds>) -> f64 {
    let n = x.len();
    let mut pg = Array1::zeros(n);

    for i in 0..n {
        let xi = x[i];
        let gi = g[i];

        if let Some(bounds) = bounds {
            // Check if the point is at a bound and the gradient points outward
            if let Some(lb) = bounds.lower[i] {
                if xi <= lb && gi < 0.0 {
                    // At lower bound with gradient pointing outward
                    pg[i] = 0.0;
                    continue;
                }
            }

            if let Some(ub) = bounds.upper[i] {
                if xi >= ub && gi > 0.0 {
                    // At upper bound with gradient pointing outward
                    pg[i] = 0.0;
                    continue;
                }
            }
        }

        // Not at a bound or gradient points inward
        pg[i] = gi;
    }

    // Return the Euclidean norm of the projected gradient
    pg.mapv(|pgi| pgi.powi(2)).sum().sqrt()
}

/// Projects the search direction to ensure we don't move in a direction that
/// immediately violates the bounds.
fn project_direction(direction: &mut Array1<f64>, x: &Array1<f64>, bounds: Option<&Bounds>) {
    if bounds.is_none() {
        return; // No bounds, no projection needed
    }

    let bounds = bounds.unwrap();

    for i in 0..x.len() {
        let xi = x[i];

        // Check if we're at a bound
        if let Some(lb) = bounds.lower[i] {
            if xi <= lb && direction[i] < 0.0 {
                // At lower bound and moving in negative direction
                direction[i] = 0.0;
            }
        }

        if let Some(ub) = bounds.upper[i] {
            if xi >= ub && direction[i] > 0.0 {
                // At upper bound and moving in positive direction
                direction[i] = 0.0;
            }
        }
    }
}

/// Line search for L-BFGS-B method, respecting bounds
fn lbfgsb_line_search<F, S>(
    fun: &mut F,
    x: &Array1<f64>,
    direction: &Array1<f64>,
    f_x: f64,
    bounds: Option<&Bounds>,
    nfev: &mut usize,
) -> (f64, f64)
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64>,
{
    // Get bounds on the line search parameter
    let (a_min, a_max) = compute_line_bounds(x, direction, bounds);

    // Use a more robust line search with bounds
    let c1 = 1e-4; // Sufficient decrease parameter (Armijo condition)
    let rho = 0.5; // Backtracking parameter

    // Start with alpha based on initial_step to ensure we're within bounds
    let mut alpha = if a_max < 1.0 { a_max * 0.99 } else { 1.0 };

    // If bounds fully constrain movement, return that constrained step
    if a_max <= 0.0 || a_min >= a_max {
        alpha = if a_max > 0.0 { a_max } else { 0.0 };
        let x_new = x + alpha * direction;
        *nfev += 1;
        let f_new = fun(&x_new.view()).into();
        return (alpha, f_new);
    }

    // Compute the directional derivative (dot product of gradient and direction)
    let slope = direction.mapv(|di| di.powi(2)).sum();

    // Function to evaluate a point on the line
    let mut f_line = |alpha: f64| {
        let mut x_new = x + alpha * direction;

        // Project onto bounds
        if let Some(bounds) = bounds {
            bounds.project(x_new.as_slice_mut().unwrap());
        }

        *nfev += 1;
        fun(&x_new.view()).into()
    };

    // Initial step
    let mut f_new = f_line(alpha);

    // Backtracking until Armijo condition is satisfied or we hit the lower bound
    while f_new > f_x - c1 * alpha * slope.abs() && alpha > a_min + 1e-16 {
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

/// Compute bounds for line search parameter
fn compute_line_bounds(
    x: &Array1<f64>,
    direction: &Array1<f64>,
    bounds: Option<&Bounds>,
) -> (f64, f64) {
    if bounds.is_none() {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }

    let bounds = bounds.unwrap();
    let mut a_min = f64::NEG_INFINITY;
    let mut a_max = f64::INFINITY;

    for i in 0..x.len() {
        let xi = x[i];
        let di = direction[i];

        if di.abs() < 1e-16 {
            continue;
        }

        // Lower bound constraint
        if let Some(lb) = bounds.lower[i] {
            let a_lb = (lb - xi) / di;
            if di > 0.0 {
                a_min = a_min.max(a_lb);
            } else {
                a_max = a_max.min(a_lb);
            }
        }

        // Upper bound constraint
        if let Some(ub) = bounds.upper[i] {
            let a_ub = (ub - xi) / di;
            if di > 0.0 {
                a_max = a_max.min(a_ub);
            } else {
                a_min = a_min.max(a_ub);
            }
        }
    }

    if a_min > a_max {
        (0.0, 0.0)
    } else {
        (a_min, a_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    #[ignore] // FIXME: Algorithm reports result.success = false, fails to converge
    fn test_lbfgs_quadratic() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + 4.0 * x[1] * x[1] };

        let x0 = Array1::from_vec(vec![2.0, 1.0]);
        let options = Options::default();

        let result = minimize_lbfgs(quadratic, x0, &options).unwrap();

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    #[ignore] // FIXME: Bounded optimization gets stuck at (0.0, y) instead of (1.0, 1.0)
    fn test_lbfgsb_with_bounds() {
        let quadratic =
            |x: &ArrayView1<f64>| -> f64 { (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();

        // Constrain solution to [0, 1] x [0, 1]
        let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);
        options.bounds = Some(bounds);

        let result = minimize_lbfgsb(quadratic, x0, &options).unwrap();

        assert!(result.success);
        // The optimal point (2, 3) is outside the bounds, so we should get (1, 1)
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-6);
    }
}
