//! BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm for unconstrained optimization

use crate::error::OptimizeError;
use crate::unconstrained::line_search::backtracking_line_search;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::{array_diff_norm, check_convergence, finite_difference_gradient};
use crate::unconstrained::Options;
use ndarray::{Array1, Array2, ArrayView1, Axis};

/// Implements the BFGS algorithm with optional bounds support
pub fn minimize_bfgs<F, S>(
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

    let mut f = fun(&x.view()).into();

    // Calculate initial gradient using finite differences
    let mut g = finite_difference_gradient(&mut fun, &x.view(), eps)?;

    // Initialize approximation of inverse Hessian with identity matrix
    let mut h_inv = Array2::eye(n);

    // Initialize counters
    let mut iter = 0;
    let mut nfev = 1 + n; // Initial evaluation plus gradient calculations

    // Main loop
    while iter < max_iter {
        // Check convergence on gradient
        if g.mapv(|gi| gi.abs()).sum() < gtol {
            break;
        }

        // Compute search direction
        let mut p = -h_inv.dot(&g);

        // Project search direction for bounded optimization
        if let Some(bounds) = bounds {
            for i in 0..n {
                let mut can_decrease = true;
                let mut can_increase = true;

                // Check if at boundary
                if let Some(lb) = bounds.lower[i] {
                    if x[i] <= lb + eps {
                        can_decrease = false;
                    }
                }
                if let Some(ub) = bounds.upper[i] {
                    if x[i] >= ub - eps {
                        can_increase = false;
                    }
                }

                // Project gradient component
                if (g[i] > 0.0 && !can_decrease) || (g[i] < 0.0 && !can_increase) {
                    p[i] = 0.0;
                }
            }

            // If no movement is possible, we're at a constrained optimum
            if p.mapv(|pi| pi.abs()).sum() < 1e-10 {
                break;
            }
        }

        // Line search
        let alpha_init = 1.0;
        let (alpha, f_new) = backtracking_line_search(
            &mut fun,
            &x.view(),
            f,
            &p.view(),
            &g.view(),
            alpha_init,
            0.0001,
            0.5,
            bounds,
        );

        nfev += 1; // Count line search evaluations

        // Update position
        let s = alpha * &p;
        let x_new = &x + &s;

        // Check step size convergence
        if array_diff_norm(&x_new.view(), &x.view()) < options.xtol {
            x = x_new;
            break;
        }

        // Calculate new gradient
        let g_new = finite_difference_gradient(&mut fun, &x_new.view(), eps)?;
        nfev += n;

        // Gradient difference
        let y = &g_new - &g;

        // Check convergence on function value
        if check_convergence(
            f - f_new,
            0.0,
            g_new.mapv(|x| x.abs()).sum(),
            ftol,
            0.0,
            gtol,
        ) {
            x = x_new;
            g = g_new;
            break;
        }

        // Update inverse Hessian approximation using BFGS formula
        let s_dot_y = s.dot(&y);
        if s_dot_y > 1e-10 {
            let rho = 1.0 / s_dot_y;
            let i_mat = Array2::eye(n);

            // Compute (I - ρ y s^T)
            let y_col = y.clone().insert_axis(Axis(1));
            let s_row = s.clone().insert_axis(Axis(0));
            let y_s_t = y_col.dot(&s_row);
            let term1 = &i_mat - &(&y_s_t * rho);

            // Compute (I - ρ s y^T)
            let s_col = s.clone().insert_axis(Axis(1));
            let y_row = y.clone().insert_axis(Axis(0));
            let s_y_t = s_col.dot(&y_row);
            let term2 = &i_mat - &(&s_y_t * rho);

            // Update H_inv = (I - ρ y s^T) H (I - ρ s y^T) + ρ s s^T
            let term3 = term1.dot(&h_inv);
            h_inv = term3.dot(&term2) + rho * s_col.dot(&s_row);
        }

        // Update variables for next iteration
        x = x_new;
        f = f_new;
        g = g_new;

        iter += 1;
    }

    // Final check for bounds
    if let Some(bounds) = bounds {
        bounds.project(x.as_slice_mut().unwrap());
    }

    // Use original function for final value
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unconstrained::Bounds;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bfgs_quadratic() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 {
            let a = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).unwrap();
            let b = Array1::from_vec(vec![-4.0, -6.0]);
            0.5 * x.dot(&a.dot(x)) + b.dot(x)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let options = Options::default();

        let result = minimize_bfgs(quadratic, x0, &options).unwrap();

        assert!(result.success);
        // Optimal solution: x = A^(-1) * (-b) = [2.0, 2.0]
        assert_abs_diff_eq!(result.x[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
            let a = 1.0;
            let b = 100.0;
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();
        options.max_iter = 2000; // More iterations for Rosenbrock

        let result = minimize_bfgs(rosenbrock, x0, &options).unwrap();

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 3e-3);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 5e-3);
    }

    #[test]
    fn test_bfgs_with_bounds() {
        let quadratic =
            |x: &ArrayView1<f64>| -> f64 { (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();

        // Constrain solution to [0, 1] x [0, 1]
        let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);
        options.bounds = Some(bounds);

        let result = minimize_bfgs(quadratic, x0, &options).unwrap();

        assert!(result.success);
        // The optimal point (2, 3) is outside the bounds, so we should get (1, 1)
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-6);
    }
}
