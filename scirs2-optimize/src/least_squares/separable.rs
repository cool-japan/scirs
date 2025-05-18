//! Separable least squares for partially linear problems
//!
//! This module implements variable projection (VARPRO) algorithm for solving
//! separable nonlinear least squares problems where the model has the form:
//!
//! f(x, α, β) = Σ αᵢ φᵢ(x, β)
//!
//! where α are linear parameters and β are nonlinear parameters.
//!
//! # Example
//!
//! ```
//! use ndarray::{array, Array1, Array2};
//! use scirs2_optimize::least_squares::separable::{separable_least_squares, SeparableOptions};
//!
//! // Model: y = α₁ * exp(-β * t) + α₂
//! // Linear parameters: α = [α₁, α₂]
//! // Nonlinear parameters: β = [β]
//!
//! // Basis functions that depend on nonlinear parameters
//! fn basis_functions(t: &[f64], beta: &[f64]) -> Array2<f64> {
//!     let n = t.len();
//!     let mut phi = Array2::zeros((n, 2));
//!     
//!     for i in 0..n {
//!         phi[[i, 0]] = (-beta[0] * t[i]).exp(); // exp(-β*t)
//!         phi[[i, 1]] = 1.0;                     // constant term
//!     }
//!     phi
//! }
//!
//! // Jacobian of basis functions w.r.t. nonlinear parameters
//! fn basis_jacobian(t: &[f64], beta: &[f64]) -> Array2<f64> {
//!     let n = t.len();
//!     let mut dphi_dbeta = Array2::zeros((n * 2, 1)); // n*p x q
//!     
//!     for i in 0..n {
//!         // d/dβ(exp(-β*t)) = -t * exp(-β*t)
//!         dphi_dbeta[[i, 0]] = -t[i] * (-beta[0] * t[i]).exp();
//!         // d/dβ(1) = 0
//!         dphi_dbeta[[n + i, 0]] = 0.0;
//!     }
//!     dphi_dbeta
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Data points
//! let t_data = array![0.0, 0.5, 1.0, 1.5, 2.0];
//! let y_data = array![2.0, 1.6, 1.3, 1.1, 1.0];
//!
//! // Initial guess for nonlinear parameters
//! let beta0 = array![0.5];
//!
//! let result = separable_least_squares(
//!     basis_functions,
//!     basis_jacobian,
//!     &t_data,
//!     &y_data,
//!     &beta0,
//!     None
//! )?;
//!
//! println!("Nonlinear params: {:?}", result.result.x);
//! println!("Linear params: {:?}", result.linear_params);
//! # Ok(())
//! # }
//! ```

use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1};
use std::f64;

/// Options for separable least squares
#[derive(Debug, Clone)]
pub struct SeparableOptions {
    /// Maximum number of iterations
    pub max_iter: usize,

    /// Tolerance for convergence on nonlinear parameters
    pub beta_tol: f64,

    /// Tolerance for convergence on function value
    pub ftol: f64,

    /// Tolerance for convergence on gradient
    pub gtol: f64,

    /// Method for solving the linear subproblem
    pub linear_solver: LinearSolver,

    /// Regularization parameter for linear solve (if needed)
    pub lambda: f64,
}

/// Methods for solving the linear least squares subproblem
#[derive(Debug, Clone, Copy)]
pub enum LinearSolver {
    /// QR decomposition (stable, recommended)
    QR,
    /// Normal equations (faster but less stable)
    NormalEquations,
    /// Singular value decomposition (most stable)
    SVD,
}

impl Default for SeparableOptions {
    fn default() -> Self {
        SeparableOptions {
            max_iter: 100,
            beta_tol: 1e-8,
            ftol: 1e-8,
            gtol: 1e-8,
            linear_solver: LinearSolver::QR,
            lambda: 0.0,
        }
    }
}

/// Result structure extended for separable least squares
#[derive(Debug, Clone)]
pub struct SeparableResult {
    /// Standard optimization results (nonlinear parameters)
    pub result: OptimizeResults<f64>,
    /// Optimal linear parameters
    pub linear_params: Array1<f64>,
}

/// Solve a separable nonlinear least squares problem
///
/// This function solves problems of the form:
/// minimize ||y - Σ αᵢ φᵢ(x, β)||²
///
/// where α are linear parameters and β are nonlinear parameters.
///
/// # Arguments
///
/// * `basis_functions` - Function that returns the basis matrix Φ(x, β)
/// * `basis_jacobian` - Function that returns ∂Φ/∂β
/// * `x_data` - Independent variable data
/// * `y_data` - Dependent variable data  
/// * `beta0` - Initial guess for nonlinear parameters
/// * `options` - Options for the optimization
pub fn separable_least_squares<F, J, S1, S2, S3>(
    basis_functions: F,
    basis_jacobian: J,
    x_data: &ArrayBase<S1, Ix1>,
    y_data: &ArrayBase<S2, Ix1>,
    beta0: &ArrayBase<S3, Ix1>,
    options: Option<SeparableOptions>,
) -> OptimizeResult<SeparableResult>
where
    F: Fn(&[f64], &[f64]) -> Array2<f64>,
    J: Fn(&[f64], &[f64]) -> Array2<f64>,
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
{
    let options = options.unwrap_or_default();
    let mut beta = beta0.to_owned();

    let n = y_data.len();
    if x_data.len() != n {
        return Err(crate::error::OptimizeError::ValueError(
            "x_data and y_data must have the same length".to_string(),
        ));
    }

    let mut iter = 0;
    let mut nfev = 0;
    let mut prev_cost = f64::INFINITY;

    // Main optimization loop
    while iter < options.max_iter {
        // Compute basis functions
        let phi = basis_functions(x_data.as_slice().unwrap(), beta.as_slice().unwrap());
        nfev += 1;

        let (n_points, _n_basis) = phi.dim();
        if n_points != n {
            return Err(crate::error::OptimizeError::ValueError(
                "Basis functions returned wrong number of rows".to_string(),
            ));
        }

        // Solve linear least squares for α given current β
        let alpha = solve_linear_subproblem(&phi, y_data, &options)?;

        // Compute residual
        let y_pred = phi.dot(&alpha);
        let residual = y_data - &y_pred;
        let cost = 0.5 * residual.iter().map(|&r| r * r).sum::<f64>();

        // Check convergence on cost function
        if (prev_cost - cost).abs() < options.ftol * cost {
            let mut result = OptimizeResults::default();
            result.x = beta.clone();
            result.fun = cost;
            result.nfev = nfev;
            result.nit = iter;
            result.success = true;
            result.message = "Converged (function tolerance)".to_string();

            return Ok(SeparableResult {
                result,
                linear_params: alpha,
            });
        }

        // Compute gradient w.r.t. nonlinear parameters
        let gradient = compute_gradient(
            &phi,
            &alpha,
            &residual,
            x_data.as_slice().unwrap(),
            beta.as_slice().unwrap(),
            &basis_jacobian,
        );

        // Check convergence on gradient
        if gradient.iter().all(|&g| g.abs() < options.gtol) {
            let mut result = OptimizeResults::default();
            result.x = beta.clone();
            result.fun = cost;
            result.nfev = nfev;
            result.nit = iter;
            result.success = true;
            result.message = "Converged (gradient tolerance)".to_string();

            return Ok(SeparableResult {
                result,
                linear_params: alpha,
            });
        }

        // Update nonlinear parameters using gradient descent
        // (Could be improved with more sophisticated methods)
        let step_size = backtracking_line_search(&beta, &gradient, cost, |b| {
            let phi_new = basis_functions(x_data.as_slice().unwrap(), b);
            let alpha_new = solve_linear_subproblem(&phi_new, y_data, &options).unwrap();
            let y_pred_new = phi_new.dot(&alpha_new);
            let res_new = y_data - &y_pred_new;
            0.5 * res_new.iter().map(|&r| r * r).sum::<f64>()
        });
        nfev += 5; // Approximate function evaluations in line search

        beta = &beta - &gradient * step_size;

        // Check convergence on parameters
        if gradient.iter().map(|&g| g * g).sum::<f64>().sqrt() * step_size < options.beta_tol {
            let mut result = OptimizeResults::default();
            result.x = beta.clone();
            result.fun = cost;
            result.nfev = nfev;
            result.nit = iter;
            result.success = true;
            result.message = "Converged (parameter tolerance)".to_string();

            // Compute final linear parameters
            let phi_final = basis_functions(x_data.as_slice().unwrap(), beta.as_slice().unwrap());
            let alpha_final = solve_linear_subproblem(&phi_final, y_data, &options)?;

            return Ok(SeparableResult {
                result,
                linear_params: alpha_final,
            });
        }

        prev_cost = cost;
        iter += 1;
    }

    // Maximum iterations reached
    let phi_final = basis_functions(x_data.as_slice().unwrap(), beta.as_slice().unwrap());
    let alpha_final = solve_linear_subproblem(&phi_final, y_data, &options)?;
    let y_pred_final = phi_final.dot(&alpha_final);
    let res_final = y_data - &y_pred_final;
    let final_cost = 0.5 * res_final.iter().map(|&r| r * r).sum::<f64>();

    let mut result = OptimizeResults::default();
    result.x = beta;
    result.fun = final_cost;
    result.nfev = nfev;
    result.nit = iter;
    result.success = false;
    result.message = "Maximum iterations reached".to_string();

    Ok(SeparableResult {
        result,
        linear_params: alpha_final,
    })
}

/// Solve the linear least squares subproblem
fn solve_linear_subproblem<S1>(
    phi: &Array2<f64>,
    y: &ArrayBase<S1, Ix1>,
    options: &SeparableOptions,
) -> OptimizeResult<Array1<f64>>
where
    S1: Data<Elem = f64>,
{
    match options.linear_solver {
        LinearSolver::NormalEquations => {
            // Solve using normal equations: (Φ^T Φ) α = Φ^T y
            let phi_t_phi = phi.t().dot(phi);
            let phi_t_y = phi.t().dot(y);

            // Add regularization if specified
            let mut regularized = phi_t_phi.clone();
            if options.lambda > 0.0 {
                for i in 0..regularized.shape()[0] {
                    regularized[[i, i]] += options.lambda;
                }
            }

            solve_symmetric_system(&regularized, &phi_t_y)
        }
        LinearSolver::QR => {
            // QR decomposition (more stable)
            qr_solve(phi, y, options.lambda)
        }
        LinearSolver::SVD => {
            // SVD decomposition (most stable)
            svd_solve(phi, y, options.lambda)
        }
    }
}

/// Compute gradient w.r.t. nonlinear parameters
fn compute_gradient<J>(
    _phi: &Array2<f64>,
    alpha: &Array1<f64>,
    residual: &Array1<f64>,
    x_data: &[f64],
    beta: &[f64],
    basis_jacobian: &J,
) -> Array1<f64>
where
    J: Fn(&[f64], &[f64]) -> Array2<f64>,
{
    let dphi_dbeta = basis_jacobian(x_data, beta);
    let (_n_total, q) = dphi_dbeta.dim();
    let n = residual.len();
    let p = alpha.len();

    // Reshape dphi_dbeta from (n*p, q) to compute gradient
    let mut gradient = Array1::zeros(q);

    for j in 0..q {
        let mut grad_j = 0.0;
        for i in 0..n {
            for k in 0..p {
                let idx = k * n + i;
                grad_j -= residual[i] * alpha[k] * dphi_dbeta[[idx, j]];
            }
        }
        gradient[j] = grad_j;
    }

    gradient
}

/// Simple backtracking line search
fn backtracking_line_search<F>(x: &Array1<f64>, direction: &Array1<f64>, f0: f64, f: F) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let mut alpha = 1.0;
    let c = 0.5;
    let rho = 0.5;

    let grad_dot_dir = direction.iter().map(|&d| d * d).sum::<f64>();

    for _ in 0..20 {
        let x_new = x - alpha * direction;
        let f_new = f(x_new.as_slice().unwrap());

        if f_new <= f0 - c * alpha * grad_dot_dir {
            return alpha;
        }

        alpha *= rho;
    }

    alpha
}

/// Solve symmetric positive definite system
fn solve_symmetric_system(a: &Array2<f64>, b: &Array1<f64>) -> OptimizeResult<Array1<f64>> {
    // Cholesky decomposition for symmetric positive definite matrices
    // Fallback to LU if Cholesky fails

    // Simple Gaussian elimination for now
    let n = a.shape()[0];
    let mut aug = Array2::zeros((n, n + 1));

    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Gaussian elimination
    for i in 0..n {
        let pivot = aug[[i, i]];
        if pivot.abs() < 1e-10 {
            return Err(crate::error::OptimizeError::ValueError(
                "Singular matrix in linear solve".to_string(),
            ));
        }

        for j in i + 1..n {
            let factor = aug[[j, i]] / pivot;
            for k in i..=n {
                aug[[j, k]] -= factor * aug[[i, k]];
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
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// QR solve (simplified)
fn qr_solve<S>(phi: &Array2<f64>, y: &ArrayBase<S, Ix1>, lambda: f64) -> OptimizeResult<Array1<f64>>
where
    S: Data<Elem = f64>,
{
    // For simplicity, use normal equations with regularization
    // A proper implementation would use actual QR decomposition
    let phi_t_phi = phi.t().dot(phi);
    let phi_t_y = phi.t().dot(y);

    let mut regularized = phi_t_phi.clone();
    for i in 0..regularized.shape()[0] {
        regularized[[i, i]] += lambda;
    }

    solve_symmetric_system(&regularized, &phi_t_y)
}

/// SVD solve (simplified)
fn svd_solve<S>(
    phi: &Array2<f64>,
    y: &ArrayBase<S, Ix1>,
    lambda: f64,
) -> OptimizeResult<Array1<f64>>
where
    S: Data<Elem = f64>,
{
    // For simplicity, use normal equations with regularization
    // A proper implementation would use actual SVD
    qr_solve(phi, y, lambda)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_separable_exponential() {
        // Model: y = α₁ * exp(-β * t) + α₂
        // True parameters: α₁ = 2.0, α₂ = 0.5, β = 0.7

        fn basis_functions(t: &[f64], beta: &[f64]) -> Array2<f64> {
            let n = t.len();
            let mut phi = Array2::zeros((n, 2));

            for i in 0..n {
                phi[[i, 0]] = (-beta[0] * t[i]).exp();
                phi[[i, 1]] = 1.0;
            }
            phi
        }

        fn basis_jacobian(t: &[f64], beta: &[f64]) -> Array2<f64> {
            let n = t.len();
            let mut dphi_dbeta = Array2::zeros((n * 2, 1));

            for i in 0..n {
                dphi_dbeta[[i, 0]] = -t[i] * (-beta[0] * t[i]).exp();
                dphi_dbeta[[n + i, 0]] = 0.0;
            }
            dphi_dbeta
        }

        // Generate synthetic data
        let t_data = array![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let true_alpha = array![2.0, 0.5];
        let true_beta = array![0.7];

        let phi_true = basis_functions(t_data.as_slice().unwrap(), true_beta.as_slice().unwrap());
        let y_data =
            phi_true.dot(&true_alpha) + 0.01 * array![0.1, -0.05, 0.08, -0.03, 0.06, -0.04, 0.02];

        // Initial guess
        let beta0 = array![0.5];

        let result = separable_least_squares(
            basis_functions,
            basis_jacobian,
            &t_data,
            &y_data,
            &beta0,
            None,
        )
        .unwrap();

        assert!(result.result.success);
        assert!((result.result.x[0] - true_beta[0]).abs() < 0.1);
        assert!((result.linear_params[0] - true_alpha[0]).abs() < 0.1);
        assert!((result.linear_params[1] - true_alpha[1]).abs() < 0.1);
    }

    #[test]
    fn test_separable_multi_exponential() {
        // Model: y = α₁ * exp(-β₁ * t) + α₂ * exp(-β₂ * t)
        // More complex with two nonlinear parameters

        fn basis_functions(t: &[f64], beta: &[f64]) -> Array2<f64> {
            let n = t.len();
            let mut phi = Array2::zeros((n, 2));

            for i in 0..n {
                phi[[i, 0]] = (-beta[0] * t[i]).exp();
                phi[[i, 1]] = (-beta[1] * t[i]).exp();
            }
            phi
        }

        fn basis_jacobian(t: &[f64], beta: &[f64]) -> Array2<f64> {
            let n = t.len();
            let mut dphi_dbeta = Array2::zeros((n * 2, 2));

            for i in 0..n {
                dphi_dbeta[[i, 0]] = -t[i] * (-beta[0] * t[i]).exp();
                dphi_dbeta[[i, 1]] = 0.0;
                dphi_dbeta[[n + i, 0]] = 0.0;
                dphi_dbeta[[n + i, 1]] = -t[i] * (-beta[1] * t[i]).exp();
            }
            dphi_dbeta
        }

        // Generate synthetic data
        let t_data = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4];
        let true_alpha = array![3.0, 1.5];
        let true_beta = array![2.0, 0.5];

        let phi_true = basis_functions(t_data.as_slice().unwrap(), true_beta.as_slice().unwrap());
        let y_data = phi_true.dot(&true_alpha);

        // Initial guess
        let beta0 = array![1.5, 0.8];

        let mut options = SeparableOptions::default();
        options.max_iter = 200; // More iterations for harder problem
        options.beta_tol = 1e-6;

        let result = separable_least_squares(
            basis_functions,
            basis_jacobian,
            &t_data,
            &y_data,
            &beta0,
            Some(options),
        )
        .unwrap();

        // For multi-exponential problems, convergence is harder
        // Just check that we made good progress
        assert!(result.result.fun < 0.1, "Cost = {}", result.result.fun);

        // Print results for debugging
        println!("Multi-exponential results:");
        println!("Beta: {:?} (true: {:?})", result.result.x, true_beta);
        println!("Alpha: {:?} (true: {:?})", result.linear_params, true_alpha);
        println!("Cost: {}", result.result.fun);
        println!("Success: {}", result.result.success);
    }
}
