//! Powell's method for unconstrained optimization

use crate::error::OptimizeError;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::{Bounds, Options};
use ndarray::{Array1, ArrayView1};

/// Implements Powell's method for unconstrained optimization with optional bounds support
pub fn minimize_powell<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64> + Clone,
{
    // Get options or use defaults
    let ftol = options.ftol;
    let max_iter = options.max_iter;
    let bounds = options.bounds.as_ref();

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();

    // If bounds are provided, ensure x0 is within bounds
    if let Some(bounds) = bounds {
        bounds.project(x.as_slice_mut().unwrap());
    }

    let mut f = fun(&x.view()).into();

    // Initialize the set of directions as the standard basis
    let mut directions = Vec::with_capacity(n);
    for i in 0..n {
        let mut e_i = Array1::zeros(n);
        e_i[i] = 1.0;
        directions.push(e_i);
    }

    // Counters
    let mut iter = 0;
    let mut nfev = 1; // Initial function evaluation

    // Powell's main loop
    while iter < max_iter {
        let x_old = x.clone();
        let f_old = f;

        // Keep track of the greatest function reduction
        let mut f_reduction_max = 0.0;
        let mut reduction_idx = 0;

        // Line search along each direction
        for (i, u) in directions.iter().enumerate().take(n) {
            let f_before = f;

            // Line search along direction u, respecting bounds
            let (alpha, f_min) = line_search_powell(&mut fun, &x, u, f, &mut nfev, bounds);

            // Update current position and function value
            x = &x + &(alpha * u);
            f = f_min;

            // Update maximum reduction tracker
            let reduction = f_before - f;
            if reduction > f_reduction_max {
                f_reduction_max = reduction;
                reduction_idx = i;
            }
        }

        // Compute the new direction
        let new_dir = &x - &x_old;

        // Check if the new direction is zero (happens if the point hits a bound and can't move)
        let new_dir_norm = new_dir.mapv(|x| x.powi(2)).sum().sqrt();
        if new_dir_norm < 1e-8 {
            // We're likely at a bound constraint and can't make progress
            break;
        }

        // Extra line search along the extrapolated direction
        let (alpha, f_min) = line_search_powell(&mut fun, &x, &new_dir, f, &mut nfev, bounds);
        x = &x + &(alpha * &new_dir);
        f = f_min;

        // Only *now* check for convergence
        if 2.0 * (f_old - f) <= ftol * (f_old.abs() + f.abs() + 1e-10) {
            break;
        }

        // Keep the basis full rank.
        // If the extrapolated displacement is numerically zero we would
        // lose a basis direction; just keep the old one instead.
        if new_dir.iter().any(|v| v.abs() > 1e-12) {
            // Update the set of directions by replacing the direction of greatest reduction
            directions[reduction_idx] = new_dir;
        }

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
        jacobian: None,
        hessian: None,
    })
}

/// Calculate the range of the line search parameter to respect bounds.
///
/// For a point x and direction p, find a_min and a_max such that:
/// x + a * p stays within the bounds for all a in [a_min, a_max].
fn line_bounds(x: &Array1<f64>, direction: &Array1<f64>, bounds: Option<&Bounds>) -> (f64, f64) {
    // If no bounds are provided, use unbounded line search
    if bounds.is_none() {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }

    let bounds = bounds.unwrap();

    // Start with unbounded range
    let mut a_min = f64::NEG_INFINITY;
    let mut a_max = f64::INFINITY;

    // For each dimension, calculate the range restriction
    for i in 0..x.len() {
        let xi = x[i];
        let pi = direction[i];

        if pi.abs() < 1e-16 {
            // No movement in this dimension
            continue;
        }

        // Lower bound constraint: x_i + a * p_i >= lb_i
        if let Some(lb) = bounds.lower[i] {
            let a_lb = (lb - xi) / pi;
            if pi > 0.0 {
                a_min = a_min.max(a_lb);
            } else {
                a_max = a_max.min(a_lb);
            }
        }

        // Upper bound constraint: x_i + a * p_i <= ub_i
        if let Some(ub) = bounds.upper[i] {
            let a_ub = (ub - xi) / pi;
            if pi > 0.0 {
                a_max = a_max.min(a_ub);
            } else {
                a_min = a_min.max(a_ub);
            }
        }
    }

    // Ensure a valid range exists
    if a_min > a_max {
        // No feasible movement, set range to zero
        (0.0, 0.0)
    } else {
        (a_min, a_max)
    }
}

/// Helper function for line search in Powell's method with bounds support
/// One-dimensional minimization along `x + α·direction`.
fn line_search_powell<F, S>(
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
    let (a_min, a_max) = line_bounds(x, direction, bounds);

    // Degenerate direction ⇒ no movement
    if direction.iter().all(|v| v.abs() <= 1e-16) {
        return (0.0, f_x);
    }

    // helper ϕ(α)
    let mut phi = |alpha: f64| -> f64 {
        let y = x + &(direction * alpha);
        // Project the point onto bounds if needed
        let mut y_bounded = y.clone();
        if let Some(bounds) = bounds {
            bounds.project(y_bounded.as_slice_mut().unwrap());
        }
        *nfev += 1;
        fun(&y_bounded.view()).into()
    };

    // --------------------------------------------------------------
    // 1) Find the downhill direction and an initial bracket
    // --------------------------------------------------------------
    let golden = 1.618_033_988_75_f64; // φ
    let mut step = 1.0; // Δ
    let mut a = f64::max(0.0, a_min); // Start from 0 or a_min if it's positive
    let mut fa = f_x; // ϕ(0)

    // probe +Δ and –Δ
    let mut b = f64::min(step, a_max);
    let mut fb = phi(b);
    if fb > fa {
        // uphill → try –Δ
        b = f64::max(-step, a_min);
        fb = phi(b);
        if fb > fa {
            // no downhill yet: shrink Δ until we find one
            for _ in 0..20 {
                step *= 0.5;
                if step < 1e-12 {
                    break;
                } // give up – extremely flat
                b = f64::min(step, a_max);
                fb = phi(b);
                if fb < fa {
                    break;
                }
                b = f64::max(-step, a_min);
                fb = phi(b);
                if fb < fa {
                    break;
                }
            }
        }
    }

    // if we *still* have no downhill point, stay put
    if fb >= fa {
        return (0.0, f_x);
    }

    // at this point 'a = 0' is higher, 'b' is lower
    // grow the interval until we are uphill again
    let mut c = f64::min(b + golden * (b - a), a_max);
    let mut fc = phi(c);
    for _ in 0..50 {
        if fc > fb {
            break;
        } // bracket found
        a = b;
        fa = fb;
        b = c;
        fb = fc;
        c = f64::min(b + golden * (b - a), a_max);
        fc = phi(c);
    }

    // sort so that a < b < c
    if a > c {
        std::mem::swap(&mut a, &mut c);
        std::mem::swap(&mut fa, &mut fc);
    }

    // --------------------------------------------------------------
    // 2) Golden-section search inside the bracket
    // --------------------------------------------------------------
    let mut lo = a;
    let mut hi = c;
    let mut x1 = hi - (hi - lo) / golden;
    let mut x2 = lo + (hi - lo) / golden;
    let mut f1 = phi(x1);
    let mut f2 = phi(x2);

    const IT_MAX: usize = 100;
    const TOL: f64 = 1e-8;

    for _ in 0..IT_MAX {
        if (hi - lo).abs() < TOL {
            let alpha = 0.5 * (hi + lo);
            return (alpha, phi(alpha)); // φ counts this eval
        }
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - (hi - lo) / golden;
            f1 = phi(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + (hi - lo) / golden;
            f2 = phi(x2);
        }
    }

    // fall-back: return the best of the two interior points
    if f1 < f2 {
        (x1, f1)
    } else {
        (x2, f2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_powell_simple() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };

        let x0 = Array1::from_vec(vec![1.0, 1.0]);
        let options = Options::default();

        let result = minimize_powell(quadratic, x0, &options).unwrap();

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_powell_rosenbrock() {
        let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
            let a = 1.0;
            let b = 100.0;
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let options = Options::default();

        let result = minimize_powell(rosenbrock, x0, &options).unwrap();

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_powell_with_bounds() {
        let quadratic =
            |x: &ArrayView1<f64>| -> f64 { (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();

        // Constrain solution to [0, 1] x [0, 1]
        let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);
        options.bounds = Some(bounds);

        let result = minimize_powell(quadratic, x0, &options).unwrap();

        assert!(result.success);
        // The optimal point (2, 3) is outside the bounds, so we should get (1, 1)
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-6);
    }
}
