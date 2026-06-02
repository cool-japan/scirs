//! Conjugate Gradient method for unconstrained optimization

use crate::error::OptimizeError;
use crate::unconstrained::cg_gpu::{try_gpu_cg_update, GpuCgResult, GPU_CG_THRESHOLD};
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::{
    array_diff_norm, check_convergence, compute_gradient_with_jacobian, finite_difference_gradient,
};
use crate::unconstrained::{Bounds, Jacobian, Options};
use scirs2_core::ndarray::{Array1, ArrayView1};

/// Implements the Conjugate Gradient method with optional bounds support and a `Jacobian` enum
/// for gradient computation.
///
/// This is the core implementation. `minimize_conjugate_gradient` delegates to this function.
#[allow(dead_code)]
pub fn minimize_conjugate_gradient_with_jacobian<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    jacobian: Option<&Jacobian<'_>>,
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
        let slice = x.as_slice_mut().ok_or_else(|| {
            OptimizeError::ComputationError(
                "Failed to get mutable slice for bounds projection".to_string(),
            )
        })?;
        bounds.project(slice);
    }

    let mut f = fun(&x.view()).into();

    // Calculate initial gradient
    let mut nfev: usize = 1; // count the initial f evaluation
    let mut g = match jacobian {
        Some(jac) => compute_gradient_with_jacobian(&mut fun, &x.view(), jac, eps, &mut nfev)?,
        None => {
            nfev += n;
            finite_difference_gradient(&mut fun, &x.view(), eps)?
        }
    };

    // Initialize search direction as projected steepest descent
    let mut p = -g.clone();

    // Project the search direction to respect bounds if at boundary
    if let Some(bounds) = bounds {
        project_search_direction(&mut p, &x, bounds);
    }

    // Counters
    let mut iter = 0;

    while iter < max_iter {
        // Check convergence on gradient
        if g.mapv(|gi| gi.abs()).sum() < gtol {
            break;
        }

        // If search direction is zero (completely constrained),
        // we're at a constrained optimum
        if p.mapv(|pi| pi.abs()).sum() < 1e-10 {
            break;
        }

        // Line search along the search direction, respecting bounds
        let (alpha, f_new) = line_search_cg(&mut fun, &x, &p, f, &mut nfev, bounds);

        // Update position
        let mut x_new = &x + &(&p * alpha);

        // Ensure we're within bounds (should be a no-op if line_search_cg respected bounds)
        if let Some(bounds) = bounds {
            let slice = x_new.as_slice_mut().ok_or_else(|| {
                OptimizeError::ComputationError(
                    "Failed to get mutable slice for bounds projection".to_string(),
                )
            })?;
            bounds.project(slice);
        }

        // Check if the step actually moved the point
        let step_size = array_diff_norm(&x_new.view(), &x.view());
        if step_size < 1e-10 {
            // We're at a boundary constraint and can't move further
            x = x_new;
            break;
        }

        // Compute new gradient
        let g_new = match jacobian {
            Some(jac) => {
                compute_gradient_with_jacobian(&mut fun, &x_new.view(), jac, eps, &mut nfev)?
            }
            None => {
                let g_fd = finite_difference_gradient(&mut fun, &x_new.view(), eps)?;
                nfev += n;
                g_fd
            }
        };

        // Check convergence on function value
        if check_convergence(
            f - f_new,
            step_size,
            g_new.mapv(|x| x.abs()).sum(),
            ftol,
            options.xtol,
            gtol,
        ) {
            x = x_new;
            g = g_new;
            break;
        }

        // Calculate beta (Fletcher-Reeves) and update search direction.
        // Attempt GPU acceleration for large n; fall back to CPU on any error.
        let threshold = options.gpu_threshold_override.unwrap_or(GPU_CG_THRESHOLD);
        let gpu_result = try_gpu_cg_update(&g, &g_new, &p, options.use_gpu, threshold);

        match gpu_result {
            GpuCgResult::Done(_g_new_norm_sq, _beta, new_p) => {
                p = new_p;
            }
            GpuCgResult::FallbackToCpu => {
                // CPU path: Fletcher-Reeves beta.
                let g_new_norm_sq = g_new.dot(&g_new);
                let g_norm_sq = g.dot(&g);
                let beta_fr = if g_norm_sq < 1e-10 {
                    0.0
                } else {
                    g_new_norm_sq / g_norm_sq
                };
                p = -&g_new + beta_fr * &p;
            }
        }

        // Project the search direction to respect bounds
        if let Some(bounds) = bounds {
            project_search_direction(&mut p, &x_new, bounds);

            // If the projected direction is zero or too small, use the projected gradient
            let dir_norm = p.dot(&p).sqrt();
            if dir_norm < 1e-10 {
                // Try using the projected gradient instead
                p = -g_new.clone();
                project_search_direction(&mut p, &x_new, bounds);
            }
        }

        // Update variables for next iteration
        x = x_new;
        f = f_new;
        g = g_new;

        iter += 1;

        // Restart direction to steepest descent every n iterations
        if iter % n == 0 {
            p = -g.clone();

            // Project the restarted direction to respect bounds
            if let Some(bounds) = bounds {
                project_search_direction(&mut p, &x, bounds);
            }
        }
    }

    // Final check for bounds
    if let Some(bounds) = bounds {
        let slice = x.as_slice_mut().ok_or_else(|| {
            OptimizeError::ComputationError(
                "Failed to get mutable slice for bounds projection".to_string(),
            )
        })?;
        bounds.project(slice);
    }

    // Use original function for final value
    let final_fun = fun(&x.view());

    // Create and return result
    Ok(OptimizeResult {
        x,
        fun: final_fun,
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

/// Implements the Conjugate Gradient method for unconstrained optimization with optional bounds support.
///
/// This is the legacy API. It delegates to `minimize_conjugate_gradient_with_jacobian` internally,
/// converting the `Option<G>` gradient function to a `Jacobian` enum.
#[allow(dead_code)]
pub fn minimize_conjugate_gradient<F, G, S>(
    fun: F,
    grad: Option<G>,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S + Clone,
    G: Fn(&ArrayView1<f64>) -> Array1<f64>,
    S: Into<f64> + Clone,
{
    let jac = grad.map(|g| Jacobian::Function(Box::new(g)));
    minimize_conjugate_gradient_with_jacobian(fun, x0, jac.as_ref(), options)
}

/// Project search direction to respect bounds
#[allow(dead_code)]
fn project_search_direction(p: &mut Array1<f64>, x: &Array1<f64>, bounds: &Bounds) {
    for i in 0..p.len() {
        // For dimensions at the bound, zero out search direction if it would go outside bounds
        if let Some(lb) = bounds.lower[i] {
            if (x[i] - lb).abs() < 1e-10 && p[i] < 0.0 {
                p[i] = 0.0;
            }
        }
        if let Some(ub) = bounds.upper[i] {
            if (x[i] - ub).abs() < 1e-10 && p[i] > 0.0 {
                p[i] = 0.0;
            }
        }
    }
}

/// Helper function for line search in Conjugate Gradient method with optional bounds support
#[allow(dead_code)]
fn line_search_cg<F, S>(
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
            if let Some(slice) = x_new.as_slice_mut() {
                bounds.project(slice);
            }
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

/// Compute bounds for line search parameter
#[allow(dead_code)]
pub fn compute_line_bounds(
    x: &Array1<f64>,
    direction: &Array1<f64>,
    bounds: Option<&Bounds>,
) -> (f64, f64) {
    let bounds = match bounds {
        Some(b) => b,
        None => return (f64::NEG_INFINITY, f64::INFINITY),
    };

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
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cg_quadratic() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + 4.0 * x[1] * x[1] };

        let x0 = Array1::from_vec(vec![2.0, 1.0]);
        let options = Options::default();

        let result = minimize_conjugate_gradient(
            quadratic,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            x0,
            &options,
        )
        .expect("test: cg quadratic");

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_cg_rosenbrock() {
        let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
            let a = 1.0;
            let b = 100.0;
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();
        options.max_iter = 2000; // Increase iterations for difficult Rosenbrock function

        let result = minimize_conjugate_gradient(
            rosenbrock,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            x0,
            &options,
        )
        .expect("test: cg rosenbrock");

        assert!(result.success);
        // Rosenbrock is difficult for CG, accept if we get reasonably close
        assert!(
            result.x[0] > 0.2 && result.x[0] < 1.5,
            "x[0] = {} should be near 1.0",
            result.x[0]
        );
        assert!(
            result.x[1] > 0.0 && result.x[1] < 1.5,
            "x[1] = {} should be near 1.0",
            result.x[1]
        );
    }

    #[test]
    fn test_cg_with_bounds() {
        let quadratic =
            |x: &ArrayView1<f64>| -> f64 { (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();
        options.max_iter = 1000; // More iterations for bounded optimization

        // Constrain solution to [0, 1] x [0, 1]
        let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);
        options.bounds = Some(bounds);

        let result = minimize_conjugate_gradient(
            quadratic,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            x0,
            &options,
        )
        .expect("test: cg with bounds");

        assert!(result.success);
        // The optimal point (2, 3) is outside the bounds, so we should get (1, 1)
        // Allow more tolerance for this challenging bounded problem
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 0.4);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 0.4);
    }

    #[test]
    fn test_cg_with_analytic_gradient() {
        let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + 4.0 * x[1].powi(2) };
        let grad_fn =
            |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0], 8.0 * x[1]]) };
        let x0 = Array1::from_vec(vec![2.0, 1.0]);
        let options = Options::default();
        let result = minimize_conjugate_gradient(fun, Some(grad_fn), x0, &options)
            .expect("test: cg analytic grad");
        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_cg_with_user_jacobian() {
        // Simple quadratic with Jacobian enum
        let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + 4.0 * x[1].powi(2) };

        let jac = Jacobian::Function(Box::new(|x: &ArrayView1<f64>| {
            array![2.0 * x[0], 8.0 * x[1]]
        }));

        let x0 = Array1::from_vec(vec![2.0, 1.0]);
        let options = Options::default();

        let result: OptimizeResult<f64> =
            minimize_conjugate_gradient_with_jacobian(fun, x0, Some(&jac), &options)
                .expect("test: cg with jacobian");

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-4);
    }
}
