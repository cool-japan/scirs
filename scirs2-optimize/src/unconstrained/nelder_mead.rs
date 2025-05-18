//! Nelder-Mead simplex algorithm for unconstrained optimization

use crate::error::OptimizeError;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::Options;
use ndarray::{Array1, ArrayView1};

/// Implements the Nelder-Mead simplex algorithm with optional bounds support
pub fn minimize_nelder_mead<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64> + Clone,
{
    // Nelder-Mead algorithm parameters
    let alpha = 1.0; // Reflection parameter
    let gamma = 2.0; // Expansion parameter
    let rho = 0.5; // Contraction parameter
    let sigma = 0.5; // Shrink parameter

    // Get bounds from options
    let bounds = options.bounds.as_ref();

    // Get the dimension of the problem
    let n = x0.len();

    // Set the maximum number of iterations
    let max_iter = options.max_iter;

    // Set the tolerance
    let ftol = options.ftol;

    // Create a function wrapper that respects bounds
    let mut bounded_fun = |x: &ArrayView1<f64>| -> f64 {
        if let Some(bounds) = bounds {
            if !bounds.is_feasible(x.as_slice().unwrap()) {
                // If the point is outside bounds, return a high value
                // to push the optimization back into the feasible region
                return f64::MAX;
            }
        }
        fun(x).into()
    };

    // Initialize the simplex
    let mut simplex = Vec::with_capacity(n + 1);
    let x0_vec = x0.to_owned();
    simplex.push((x0_vec.clone(), bounded_fun(&x0_vec.view())));

    // Create the initial simplex, ensuring all points are within bounds
    for i in 0..n {
        let mut xi = x0.to_owned();
        if xi[i] != 0.0 {
            xi[i] *= 1.05;
        } else {
            xi[i] = 0.00025;
        }

        // Project the point onto bounds if needed
        if let Some(bounds) = bounds {
            bounds.project(xi.as_slice_mut().unwrap());
        }

        simplex.push((xi.clone(), bounded_fun(&xi.view())));
    }

    let mut nfev = n + 1;

    // Sort the simplex by function value
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Iteration counter
    let mut iter = 0;

    // Main iteration loop
    while iter < max_iter {
        // Check convergence: if the difference in function values is less than the tolerance
        if (simplex[n].1 - simplex[0].1).abs() < ftol {
            break;
        }

        // Compute the centroid of all points except the worst one
        let mut xc = Array1::zeros(n);
        for item in simplex.iter().take(n) {
            xc = &xc + &item.0;
        }
        xc = &xc / n as f64;

        // Reflection: reflect the worst point through the centroid
        let mut xr: Array1<f64> = &xc + alpha * (&xc - &simplex[n].0);

        // Project the reflected point onto bounds if needed
        if let Some(bounds) = bounds {
            bounds.project(xr.as_slice_mut().unwrap());
        }

        let fxr = bounded_fun(&xr.view());
        nfev += 1;

        if fxr < simplex[0].1 {
            // If the reflected point is the best so far, try expansion
            let mut xe: Array1<f64> = &xc + gamma * (&xr - &xc);

            // Project the expanded point onto bounds if needed
            if let Some(bounds) = bounds {
                bounds.project(xe.as_slice_mut().unwrap());
            }

            let fxe = bounded_fun(&xe.view());
            nfev += 1;

            if fxe < fxr {
                // If the expanded point is better than the reflected point,
                // replace the worst point with the expanded point
                simplex[n] = (xe, fxe);
            } else {
                // Otherwise, replace the worst point with the reflected point
                simplex[n] = (xr, fxr);
            }
        } else if fxr < simplex[n - 1].1 {
            // If the reflected point is better than the second worst,
            // replace the worst point with the reflected point
            simplex[n] = (xr, fxr);
        } else {
            // Otherwise, try contraction
            let mut xc_contract: Array1<f64> = if fxr < simplex[n].1 {
                // Outside contraction
                &xc + rho * (&xr - &xc)
            } else {
                // Inside contraction
                &xc - rho * (&xc - &simplex[n].0)
            };

            // Project the contracted point onto bounds if needed
            if let Some(bounds) = bounds {
                bounds.project(xc_contract.as_slice_mut().unwrap());
            }

            let fxc_contract = bounded_fun(&xc_contract.view());
            nfev += 1;

            if fxc_contract < simplex[n].1 {
                // If the contracted point is better than the worst point,
                // replace the worst point with the contracted point
                simplex[n] = (xc_contract, fxc_contract);
            } else {
                // If all else fails, shrink the simplex towards the best point
                for i in 1..=n {
                    let mut new_point = &simplex[0].0 + sigma * (&simplex[i].0 - &simplex[0].0);

                    // Project the shrunk point onto bounds if needed
                    if let Some(bounds) = bounds {
                        bounds.project(new_point.as_slice_mut().unwrap());
                    }

                    simplex[i].0 = new_point;
                    simplex[i].1 = bounded_fun(&simplex[i].0.view());
                    nfev += 1;
                }
            }
        }

        // Resort the simplex
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        iter += 1;
    }

    // Get the best point and its function value
    let (x_best, f_best) = simplex[0].clone();

    // If f_best is MAX, the optimization failed to find a feasible point
    if f_best == f64::MAX {
        return Err(OptimizeError::ValueError(
            "Failed to find a feasible point within bounds".to_string(),
        ));
    }

    // Use original function for final value
    let final_value = fun(&x_best.view());

    // Create the result
    Ok(OptimizeResult {
        x: x_best,
        fun: final_value,
        iterations: iter,
        nit: iter,
        func_evals: nfev,
        nfev,
        success: iter < max_iter,
        message: if iter < max_iter {
            "Optimization terminated successfully".to_string()
        } else {
            "Maximum number of iterations reached".to_string()
        },
        jacobian: None,
        hessian: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unconstrained::Bounds;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_nelder_mead_simple() {
        let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
            let a = 1.0;
            let b = 100.0;
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let options = Options::default();

        let result = minimize_nelder_mead(rosenbrock, x0, &options).unwrap();

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_nelder_mead_with_bounds() {
        let quadratic =
            |x: &ArrayView1<f64>| -> f64 { (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();

        // Constrain solution to [0, 1] x [0, 1]
        let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);
        options.bounds = Some(bounds);

        let result = minimize_nelder_mead(quadratic, x0, &options).unwrap();

        assert!(result.success);
        // The optimal point (2, 3) is outside the bounds, so we should get (1, 1)
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }
}
