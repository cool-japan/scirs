//! Weighted least squares optimization
//!
//! This module provides weighted least squares methods where each residual
//! can have a different weight, allowing for handling heteroscedastic data
//! (data with varying variance).
//!
//! # Example
//!
//! ```
//! use ndarray::{array, Array1, Array2};
//! use scirs2_optimize::least_squares::weighted::{weighted_least_squares, WeightedOptions};
//!
//! // Define a function that returns the residuals
//! fn residual(x: &[f64], data: &[f64]) -> Array1<f64> {
//!     let n = data.len() / 2;
//!     let t_values = &data[0..n];
//!     let y_values = &data[n..];
//!     
//!     let mut res = Array1::zeros(n);
//!     for i in 0..n {
//!         res[i] = y_values[i] - (x[0] + x[1] * t_values[i]);
//!     }
//!     res
//! }
//!
//! // Define the Jacobian
//! fn jacobian(x: &[f64], data: &[f64]) -> Array2<f64> {
//!     let n = data.len() / 2;
//!     let t_values = &data[0..n];
//!     
//!     let mut jac = Array2::zeros((n, 2));
//!     for i in 0..n {
//!         jac[[i, 0]] = -1.0;
//!         jac[[i, 1]] = -t_values[i];
//!     }
//!     jac
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create data
//! let data = array![0.0, 1.0, 2.0, 3.0, 4.0, 0.1, 0.9, 2.1, 2.9, 4.1];
//!
//! // Define weights (higher weight = more importance)
//! let weights = array![1.0, 1.0, 1.0, 10.0, 10.0]; // Last two points have more weight
//!
//! // Initial guess
//! let x0 = array![0.0, 0.0];
//!
//! // Solve using weighted least squares
//! let result = weighted_least_squares(
//!     residual,
//!     &x0,
//!     &weights,
//!     Some(jacobian),
//!     &data,
//!     None
//! )?;
//!
//! assert!(result.success);
//! # Ok(())
//! # }
//! ```

use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1};

/// Options for weighted least squares optimization
#[derive(Debug, Clone)]
pub struct WeightedOptions {
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

    /// Step size for finite difference approximation
    pub diff_step: f64,

    /// Whether to check that weights are non-negative
    pub check_weights: bool,
}

impl Default for WeightedOptions {
    fn default() -> Self {
        WeightedOptions {
            max_iter: 100,
            max_nfev: None,
            xtol: 1e-8,
            ftol: 1e-8,
            gtol: 1e-8,
            diff_step: 1e-8,
            check_weights: true,
        }
    }
}

/// Solve a weighted least squares problem
///
/// This function minimizes the weighted sum of squares of residuals:
/// `sum(weights[i] * residuals[i]^2)`
///
/// # Arguments
///
/// * `residuals` - Function that returns the residuals
/// * `x0` - Initial guess for the parameters
/// * `weights` - Weights for each residual (must be non-negative)
/// * `jacobian` - Optional Jacobian function
/// * `data` - Additional data to pass to residuals and jacobian
/// * `options` - Options for the optimization
///
/// # Returns
///
/// * `OptimizeResults` containing the optimization results
pub fn weighted_least_squares<F, J, D, S1, S2, S3>(
    residuals: F,
    x0: &ArrayBase<S1, Ix1>,
    weights: &ArrayBase<S2, Ix1>,
    jacobian: Option<J>,
    data: &ArrayBase<S3, Ix1>,
    options: Option<WeightedOptions>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64], &[D]) -> Array1<f64>,
    J: Fn(&[f64], &[D]) -> Array2<f64>,
    D: Clone,
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = D>,
{
    let options = options.unwrap_or_default();

    // Check weights if requested
    if options.check_weights {
        for &w in weights.iter() {
            if w < 0.0 {
                return Err(crate::error::OptimizeError::ValueError(
                    "Weights must be non-negative".to_string(),
                ));
            }
        }
    }

    // Implementation using Gauss-Newton method with weighted residuals
    weighted_gauss_newton(residuals, x0, weights, jacobian, data, &options)
}

/// Weighted Gauss-Newton implementation
fn weighted_gauss_newton<F, J, D, S1, S2, S3>(
    residuals: F,
    x0: &ArrayBase<S1, Ix1>,
    weights: &ArrayBase<S2, Ix1>,
    jacobian: Option<J>,
    data: &ArrayBase<S3, Ix1>,
    options: &WeightedOptions,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64], &[D]) -> Array1<f64>,
    J: Fn(&[f64], &[D]) -> Array2<f64>,
    D: Clone,
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = D>,
{
    let mut x = x0.to_owned();
    let m = x.len();
    let n = weights.len();

    let max_nfev = options.max_nfev.unwrap_or(options.max_iter * m * 10);
    let mut nfev = 0;
    let mut njev = 0;
    let mut iter = 0;

    // Compute square root of weights for transforming the problem
    let sqrt_weights = weights.mapv(f64::sqrt);

    // Numerical gradient helper
    let compute_numerical_jacobian =
        |x_val: &Array1<f64>, res_val: &Array1<f64>| -> (Array2<f64>, usize) {
            let eps = options.diff_step;
            let mut jac = Array2::zeros((n, m));
            let mut count = 0;

            for j in 0..m {
                let mut x_h = x_val.clone();
                x_h[j] += eps;
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

        // Compute weighted residuals
        let weighted_res = &res * &sqrt_weights;

        // Compute cost function
        let cost = 0.5 * weighted_res.iter().map(|&r| r * r).sum::<f64>();

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

        // Apply weights to Jacobian
        let mut weighted_jac = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                weighted_jac[[i, j]] = jac[[i, j]] * sqrt_weights[i];
            }
        }

        // Compute gradient: g = J^T * W * r
        let gradient = weighted_jac.t().dot(&weighted_res);

        // Check convergence on gradient
        if gradient.iter().all(|&g| g.abs() < options.gtol) {
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

        // Form normal equations: (J^T * W * J) * delta = -J^T * W * r
        let jtw_j = weighted_jac.t().dot(&weighted_jac);
        let neg_gradient = -&gradient;

        // Solve for step
        match solve_linear_system(&jtw_j, &neg_gradient) {
            Some(step) => {
                // Simple line search
                let mut alpha = 1.0;
                let mut best_cost = cost;
                let mut best_x = x.clone();

                for _ in 0..10 {
                    let x_new = &x + &step * alpha;
                    let res_new = residuals(x_new.as_slice().unwrap(), data.as_slice().unwrap());
                    nfev += 1;

                    let weighted_res_new = &res_new * &sqrt_weights;
                    let new_cost = 0.5 * weighted_res_new.iter().map(|&r| r * r).sum::<f64>();

                    if new_cost < best_cost {
                        best_cost = new_cost;
                        best_x = x_new;
                        break;
                    }

                    alpha *= 0.5;
                }

                // Check convergence on step size
                let step_norm = step.iter().map(|&s| s * s).sum::<f64>().sqrt();
                let x_norm = x.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();

                if step_norm < options.xtol * (1.0 + x_norm) {
                    let mut result = OptimizeResults::default();
                    result.x = best_x;
                    result.fun = best_cost;
                    result.nfev = nfev;
                    result.njev = njev;
                    result.nit = iter;
                    result.success = true;
                    result.message = "Converged (step size tolerance)".to_string();
                    return Ok(result);
                }

                // Check convergence on cost function
                if (cost - best_cost).abs() < options.ftol * cost {
                    let mut result = OptimizeResults::default();
                    result.x = best_x;
                    result.fun = best_cost;
                    result.nfev = nfev;
                    result.njev = njev;
                    result.nit = iter;
                    result.success = true;
                    result.message = "Converged (function tolerance)".to_string();
                    return Ok(result);
                }

                x = best_x;
            }
            None => {
                // Singular matrix, terminate
                let mut result = OptimizeResults::default();
                result.x = x;
                result.fun = cost;
                result.nfev = nfev;
                result.njev = njev;
                result.nit = iter;
                result.success = false;
                result.message = "Singular matrix in normal equations".to_string();
                return Ok(result);
            }
        }

        iter += 1;
    }

    // Max iterations reached
    let res_final = residuals(x.as_slice().unwrap(), data.as_slice().unwrap());
    let weighted_res_final = &res_final * &sqrt_weights;
    let final_cost = 0.5 * weighted_res_final.iter().map(|&r| r * r).sum::<f64>();

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

/// Simple linear system solver (same as in robust.rs)
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
    fn test_weighted_least_squares_simple() {
        // Linear regression problem
        fn residual(x: &[f64], data: &[f64]) -> Array1<f64> {
            let n = data.len() / 2;
            let t_values = &data[0..n];
            let y_values = &data[n..];

            let mut res = Array1::zeros(n);
            for i in 0..n {
                res[i] = y_values[i] - (x[0] + x[1] * t_values[i]);
            }
            res
        }

        fn jacobian(_x: &[f64], data: &[f64]) -> Array2<f64> {
            let n = data.len() / 2;
            let t_values = &data[0..n];

            let mut jac = Array2::zeros((n, 2));
            for i in 0..n {
                jac[[i, 0]] = -1.0;
                jac[[i, 1]] = -t_values[i];
            }
            jac
        }

        // Data
        let data = array![0.0, 1.0, 2.0, 3.0, 4.0, 0.1, 0.9, 2.1, 2.9, 4.1];

        // Weights - give more importance to the last two points
        let weights = array![1.0, 1.0, 1.0, 10.0, 10.0];

        let x0 = array![0.0, 0.0];

        let result =
            weighted_least_squares(residual, &x0, &weights, Some(jacobian), &data, None).unwrap();

        assert!(result.success);
        // The solution should favor the last two points more
        assert!((result.x[1] - 1.0).abs() < 0.1); // Slope close to 1.0
    }

    #[test]
    fn test_negative_weights() {
        // Simple test function
        fn residual(x: &[f64], _data: &[f64]) -> Array1<f64> {
            array![x[0] - 1.0]
        }

        let x0 = array![0.0];
        let weights = array![-1.0]; // Invalid negative weight
        let data = array![];

        let result = weighted_least_squares(
            residual,
            &x0,
            &weights,
            None::<fn(&[f64], &[f64]) -> Array2<f64>>,
            &data,
            None,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_vs_unweighted() {
        // Linear regression problem with outlier
        fn residual(x: &[f64], data: &[f64]) -> Array1<f64> {
            let n = data.len() / 2;
            let t_values = &data[0..n];
            let y_values = &data[n..];

            let mut res = Array1::zeros(n);
            for i in 0..n {
                res[i] = y_values[i] - (x[0] + x[1] * t_values[i]);
            }
            res
        }

        // Data with an outlier
        let data = array![0.0, 1.0, 2.0, 0.0, 1.0, 10.0]; // Last point is outlier

        let x0 = array![0.0, 0.0];

        // Uniform weights (essentially unweighted)
        let weights_uniform = array![1.0, 1.0, 1.0];

        // Downweight the outlier
        let weights_robust = array![1.0, 1.0, 0.1];

        let result_uniform = weighted_least_squares(
            residual,
            &x0,
            &weights_uniform,
            None::<fn(&[f64], &[f64]) -> Array2<f64>>,
            &data,
            None,
        )
        .unwrap();

        let result_robust = weighted_least_squares(
            residual,
            &x0,
            &weights_robust,
            None::<fn(&[f64], &[f64]) -> Array2<f64>>,
            &data,
            None,
        )
        .unwrap();

        // The robust solution should have a slope closer to 1.0 (true value)
        // than the uniform weight solution
        assert!((result_robust.x[1] - 1.0).abs() < (result_uniform.x[1] - 1.0).abs());
    }
}
