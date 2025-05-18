//! Bounded-variable least squares optimization
//!
//! This module provides least squares methods with box constraints on variables.
//! It implements trust-region reflective algorithm adapted for bounds.
//!
//! # Example
//!
//! ```
//! use ndarray::{array, Array1, Array2};
//! use scirs2_optimize::{Bounds, least_squares::bounded::{bounded_least_squares, BoundedOptions}};
//!
//! // Define a function that returns the residuals
//! fn residual(x: &[f64], _data: &[f64]) -> Array1<f64> {
//!     array![
//!         x[0] + 2.0 * x[1] - 2.0,
//!         x[0] - x[1] - 1.0,
//!         x[0] + x[1] - 1.5
//!     ]
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Initial guess
//! let x0 = array![0.0, 0.0];
//!
//! // Define bounds: 0 <= x[0] <= 2, -1 <= x[1] <= 1
//! let bounds = Bounds::new(&[(Some(0.0), Some(2.0)), (Some(-1.0), Some(1.0))]);
//!
//! // Solve using bounded least squares
//! let result = bounded_least_squares(
//!     residual,
//!     &x0,
//!     Some(bounds),
//!     None::<fn(&[f64], &[f64]) -> Array2<f64>>,
//!     &array![],
//!     None
//! )?;
//!
//! assert!(result.success);
//! # Ok(())
//! # }
//! ```

use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use crate::unconstrained::Bounds;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1};

/// Options for bounded least squares optimization
#[derive(Debug, Clone)]
pub struct BoundedOptions {
    /// Maximum number of iterations
    pub max_iter: usize,

    /// Maximum number of function evaluations
    pub max_nfev: Option<usize>,

    /// Tolerance for termination by the change of parameters
    pub xtol: f64,

    /// Tolerance for termination by the change of cost function
    pub ftol: f64,

    /// Tolerance for termination by the norm of gradient
    pub gtol: f64,

    /// Initial trust region radius
    pub initial_trust_radius: f64,

    /// Maximum trust region radius
    pub max_trust_radius: f64,

    /// Step size for finite difference approximation
    pub diff_step: f64,
}

impl Default for BoundedOptions {
    fn default() -> Self {
        BoundedOptions {
            max_iter: 100,
            max_nfev: None,
            xtol: 1e-8,
            ftol: 1e-8,
            gtol: 1e-8,
            initial_trust_radius: 1.0,
            max_trust_radius: 1000.0,
            diff_step: 1e-8,
        }
    }
}

/// Solve a bounded least squares problem
///
/// This function minimizes the sum of squares of residuals subject to
/// box constraints on the variables.
///
/// # Arguments
///
/// * `residuals` - Function that returns the residuals
/// * `x0` - Initial guess for the parameters
/// * `bounds` - Optional bounds on variables
/// * `jacobian` - Optional Jacobian function
/// * `data` - Additional data to pass to residuals and jacobian
/// * `options` - Options for the optimization
pub fn bounded_least_squares<F, J, D, S1, S2>(
    residuals: F,
    x0: &ArrayBase<S1, Ix1>,
    bounds: Option<Bounds>,
    jacobian: Option<J>,
    data: &ArrayBase<S2, Ix1>,
    options: Option<BoundedOptions>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64], &[D]) -> Array1<f64>,
    J: Fn(&[f64], &[D]) -> Array2<f64>,
    D: Clone,
    S1: Data<Elem = f64>,
    S2: Data<Elem = D>,
{
    let options = options.unwrap_or_default();

    // If no bounds provided, use regular least squares
    if bounds.is_none() {
        // Call regular least squares (would need to import and use it)
        // For now, proceed with unbounded algorithm
    }

    // Use trust region reflective algorithm adapted for least squares
    trust_region_reflective(residuals, x0, bounds, jacobian, data, &options)
}

/// Trust region reflective algorithm for bounded least squares
fn trust_region_reflective<F, J, D, S1, S2>(
    residuals: F,
    x0: &ArrayBase<S1, Ix1>,
    bounds: Option<Bounds>,
    jacobian: Option<J>,
    data: &ArrayBase<S2, Ix1>,
    options: &BoundedOptions,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64], &[D]) -> Array1<f64>,
    J: Fn(&[f64], &[D]) -> Array2<f64>,
    D: Clone,
    S1: Data<Elem = f64>,
    S2: Data<Elem = D>,
{
    let mut x = x0.to_owned();
    let m = x.len();

    // Project initial point to bounds
    if let Some(ref b) = bounds {
        x = project_to_bounds(&x, b);
    }

    let max_nfev = options.max_nfev.unwrap_or(options.max_iter * m * 10);
    let mut nfev = 0;
    let mut njev = 0;
    let mut iter = 0;

    let mut trust_radius = options.initial_trust_radius;
    let max_trust_radius = options.max_trust_radius;

    // Numerical gradient helper
    let compute_numerical_jacobian =
        |x_val: &Array1<f64>, res_val: &Array1<f64>| -> (Array2<f64>, usize) {
            let eps = options.diff_step;
            let n = res_val.len();
            let mut jac = Array2::zeros((n, m));
            let mut count = 0;

            for j in 0..m {
                let mut x_h = x_val.clone();
                x_h[j] += eps;

                // Project perturbed point to bounds
                if let Some(ref b) = bounds {
                    x_h = project_to_bounds(&x_h, b);
                }

                let res_h = residuals(x_h.as_slice().unwrap(), data.as_slice().unwrap());
                count += 1;

                for i in 0..n {
                    jac[[i, j]] = (res_h[i] - res_val[i]) / eps;
                }
            }

            (jac, count)
        };

    // Main optimization loop
    while iter < options.max_iter && nfev < max_nfev {
        // Compute residuals
        let res = residuals(x.as_slice().unwrap(), data.as_slice().unwrap());
        nfev += 1;
        let _n = res.len();

        // Compute cost function
        let cost = 0.5 * res.iter().map(|&r| r * r).sum::<f64>();

        // Compute Jacobian
        let (jac, _jac_evals) = match &jacobian {
            Some(jac_fn) => {
                let j = jac_fn(x.as_slice().unwrap(), data.as_slice().unwrap());
                njev += 1;
                (j, 0)
            }
            None => {
                let (j, count) = compute_numerical_jacobian(&x, &res);
                nfev += count;
                (j, count)
            }
        };

        // Compute gradient: g = J^T * r
        let gradient = jac.t().dot(&res);

        // Compute projected gradient for convergence check
        let proj_grad = compute_projected_gradient(&x, &gradient, &bounds);

        // Check convergence on projected gradient
        if proj_grad.iter().all(|&g| g.abs() < options.gtol) {
            let mut result = OptimizeResults::default();
            result.x = x;
            result.fun = cost;
            result.nfev = nfev;
            result.njev = njev;
            result.nit = iter;
            result.success = true;
            result.message = "Optimization terminated successfully.".to_string();
            return Ok(result);
        }

        // Solve trust region subproblem with bounds
        let step = solve_trust_region_bounds(&jac, &res, &gradient, trust_radius, &x, &bounds);

        // Check step size for convergence
        let step_norm = step.iter().map(|&s| s * s).sum::<f64>().sqrt();
        if step_norm < options.xtol {
            let mut result = OptimizeResults::default();
            result.x = x;
            result.fun = cost;
            result.nfev = nfev;
            result.njev = njev;
            result.nit = iter;
            result.success = true;
            result.message = "Converged (step size tolerance)".to_string();
            return Ok(result);
        }

        // Try the step
        let mut x_new = &x + &step;

        // Project to bounds
        if let Some(ref b) = bounds {
            x_new = project_to_bounds(&x_new, b);
        }

        // Evaluate at new point
        let res_new = residuals(x_new.as_slice().unwrap(), data.as_slice().unwrap());
        nfev += 1;
        let cost_new = 0.5 * res_new.iter().map(|&r| r * r).sum::<f64>();

        // Compute actual vs predicted reduction
        let actual_reduction = cost - cost_new;
        let predicted_reduction = compute_predicted_reduction(&jac, &res, &step);

        // Compute ratio
        let rho = if predicted_reduction.abs() > 1e-10 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };

        // Update trust region based on performance
        if rho < 0.25 {
            trust_radius *= 0.25;
        } else if rho > 0.75 && step_norm >= 0.9 * trust_radius {
            trust_radius = (2.0 * trust_radius).min(max_trust_radius);
        }

        // Accept or reject step
        if rho > 0.01 {
            // Check convergence on cost function
            if actual_reduction.abs() < options.ftol * cost {
                let mut result = OptimizeResults::default();
                result.x = x_new;
                result.fun = cost_new;
                result.nfev = nfev;
                result.njev = njev;
                result.nit = iter;
                result.success = true;
                result.message = "Converged (function tolerance)".to_string();
                return Ok(result);
            }

            x = x_new;
        }

        iter += 1;
    }

    // Max iterations reached
    let res_final = residuals(x.as_slice().unwrap(), data.as_slice().unwrap());
    let final_cost = 0.5 * res_final.iter().map(|&r| r * r).sum::<f64>();

    let mut result = OptimizeResults::default();
    result.x = x;
    result.fun = final_cost;
    result.nfev = nfev;
    result.njev = njev;
    result.nit = iter;
    result.success = false;
    result.message = "Maximum iterations reached".to_string();

    Ok(result)
}

/// Project point to bounds
fn project_to_bounds(x: &Array1<f64>, bounds: &Bounds) -> Array1<f64> {
    let mut x_proj = x.clone();

    for i in 0..x.len() {
        if let Some(lb) = bounds.lower[i] {
            x_proj[i] = x_proj[i].max(lb);
        }
        if let Some(ub) = bounds.upper[i] {
            x_proj[i] = x_proj[i].min(ub);
        }
    }

    x_proj
}

/// Compute projected gradient for bounded problems
fn compute_projected_gradient(
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    bounds: &Option<Bounds>,
) -> Array1<f64> {
    let mut proj_grad = gradient.clone();

    if let Some(b) = bounds {
        for i in 0..x.len() {
            // At lower bound with positive gradient
            if let Some(lb) = b.lower[i] {
                if (x[i] - lb).abs() < 1e-10 && gradient[i] > 0.0 {
                    proj_grad[i] = 0.0;
                }
            }

            // At upper bound with negative gradient
            if let Some(ub) = b.upper[i] {
                if (x[i] - ub).abs() < 1e-10 && gradient[i] < 0.0 {
                    proj_grad[i] = 0.0;
                }
            }
        }
    }

    proj_grad
}

/// Solve trust region subproblem with bounds
fn solve_trust_region_bounds(
    jac: &Array2<f64>,
    _res: &Array1<f64>,
    gradient: &Array1<f64>,
    trust_radius: f64,
    x: &Array1<f64>,
    bounds: &Option<Bounds>,
) -> Array1<f64> {
    let m = x.len();

    // Compute Gauss-Newton step
    let jt_j = jac.t().dot(jac);
    let neg_gradient = -gradient;

    // Try to solve normal equations
    let gn_step = if let Some(step) = solve_linear_system(&jt_j, &neg_gradient) {
        step
    } else {
        // Use gradient descent as fallback
        -gradient / gradient.iter().map(|&g| g * g).sum::<f64>().sqrt()
    };

    // Apply bounds constraints
    let mut step = gn_step;

    if let Some(b) = bounds {
        for i in 0..m {
            // Clip step to respect bounds
            if let Some(lb) = b.lower[i] {
                let max_step_down = x[i] - lb;
                if step[i] < -max_step_down {
                    step[i] = -max_step_down;
                }
            }

            if let Some(ub) = b.upper[i] {
                let max_step_up = ub - x[i];
                if step[i] > max_step_up {
                    step[i] = max_step_up;
                }
            }
        }
    }

    // Apply trust region constraint
    let step_norm = step.iter().map(|&s| s * s).sum::<f64>().sqrt();
    if step_norm > trust_radius {
        step *= trust_radius / step_norm;
    }

    step
}

/// Compute predicted reduction in cost
fn compute_predicted_reduction(jac: &Array2<f64>, res: &Array1<f64>, step: &Array1<f64>) -> f64 {
    let jac_step = jac.dot(step);
    let linear_term = res.dot(&jac_step);
    let quadratic_term = 0.5 * jac_step.dot(&jac_step);

    -(linear_term + quadratic_term)
}

/// Simple linear system solver (same as in other modules)
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.shape()[0];
    if n != a.shape()[1] || n != b.len() {
        return None;
    }

    // Create augmented matrix [A|b]
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Gaussian elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = aug[[i, i]].abs();

        for j in i + 1..n {
            if aug[[j, i]].abs() > max_val {
                max_row = j;
                max_val = aug[[j, i]].abs();
            }
        }

        // Check for singularity
        if max_val < 1e-10 {
            return None;
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..=n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Eliminate below
        for j in i + 1..n {
            let c = aug[[j, i]] / aug[[i, i]];
            aug[[j, i]] = 0.0;

            for k in i + 1..=n {
                aug[[j, k]] -= c * aug[[i, k]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in i + 1..n {
            sum -= aug[[i, j]] * x[j];
        }

        if aug[[i, i]].abs() < 1e-10 {
            return None;
        }

        x[i] = sum / aug[[i, i]];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_bounded_least_squares_simple() {
        // Overdetermined system with bounds
        fn residual(x: &[f64], _data: &[f64]) -> Array1<f64> {
            array![
                x[0] + 2.0 * x[1] - 2.0,
                x[0] - x[1] - 1.0,
                x[0] + x[1] - 1.5
            ]
        }

        let x0 = array![0.0, 0.0];

        // Define bounds: 0 <= x[0] <= 2, -1 <= x[1] <= 1
        let bounds = Bounds::new(&[(Some(0.0), Some(2.0)), (Some(-1.0), Some(1.0))]);

        let result = bounded_least_squares(
            residual,
            &x0,
            Some(bounds),
            None::<fn(&[f64], &[f64]) -> Array2<f64>>,
            &array![],
            None,
        )
        .unwrap();

        assert!(result.success);
        // Check that solution respects bounds
        assert!(result.x[0] >= 0.0 && result.x[0] <= 2.0);
        assert!(result.x[1] >= -1.0 && result.x[1] <= 1.0);
    }

    #[test]
    fn test_bounded_vs_unbounded() {
        // Problem where bounds affect the solution
        fn residual(x: &[f64], _data: &[f64]) -> Array1<f64> {
            array![
                x[0] - 5.0, // Wants x[0] = 5.0
                x[1] - 3.0  // Wants x[1] = 3.0
            ]
        }

        let x0 = array![0.0, 0.0];

        // Bounds that constrain the solution
        let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);

        let result = bounded_least_squares(
            residual,
            &x0,
            Some(bounds),
            None::<fn(&[f64], &[f64]) -> Array2<f64>>,
            &array![],
            None,
        )
        .unwrap();

        assert!(result.success);
        // Solution should be at the boundary
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_project_to_bounds() {
        let x = array![2.5, -0.5, 0.5];
        let bounds = Bounds::new(&[
            (Some(0.0), Some(2.0)),
            (Some(-1.0), Some(1.0)),
            (None, None),
        ]);

        let x_proj = project_to_bounds(&x, &bounds);

        assert_eq!(x_proj[0], 2.0); // Clipped to upper bound
        assert_eq!(x_proj[1], -0.5); // Within bounds
        assert_eq!(x_proj[2], 0.5); // No bounds
    }
}
