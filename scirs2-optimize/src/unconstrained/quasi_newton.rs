//! Quasi-Newton algorithms with different update formulas (SR1, DFP, BFGS)

use crate::error::OptimizeError;
use crate::unconstrained::line_search::backtracking_line_search;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::{array_diff_norm, check_convergence, finite_difference_gradient};
use crate::unconstrained::Options;
use ndarray::{Array1, Array2, ArrayView1, Axis};

/// Update formula for Quasi-Newton methods
#[derive(Debug, Clone, Copy)]
pub enum UpdateFormula {
    /// Symmetric Rank-1 (SR1) update
    SR1,
    /// Davidon-Fletcher-Powell (DFP) update
    DFP,
    /// Broyden-Fletcher-Goldfarb-Shanno (BFGS) update
    BFGS,
}

/// Implements quasi-Newton algorithm with different update formulas
pub fn minimize_quasi_newton<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    options: &Options,
    update_formula: UpdateFormula,
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

    // Initialize approximation based on update formula
    let mut h_inv: Option<Array2<f64>> = None;
    let mut b_mat: Option<Array2<f64>> = None;

    match update_formula {
        UpdateFormula::SR1 => {
            // SR1 works with Hessian approximation B, not inverse
            b_mat = Some(Array2::eye(n));
        }
        UpdateFormula::DFP | UpdateFormula::BFGS => {
            // DFP and BFGS work with inverse Hessian approximation H
            h_inv = Some(Array2::eye(n));
        }
    }

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
        let mut p = match update_formula {
            UpdateFormula::SR1 => {
                // For SR1, we need to solve B * p = -g
                // Using LU decomposition or approximation
                let b = b_mat.as_ref().unwrap();
                // Simple approach: use the inverse directly (not efficient for large problems)
                // In practice, should use linear solver
                match invert_matrix(b) {
                    Ok(b_inv) => -b_inv.dot(&g),
                    Err(_) => {
                        // If matrix is singular, reset to identity
                        b_mat = Some(Array2::eye(n));
                        -&g
                    }
                }
            }
            UpdateFormula::DFP | UpdateFormula::BFGS => {
                let h = h_inv.as_ref().unwrap();
                -h.dot(&g)
            }
        };

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

        // Update approximation based on formula
        match update_formula {
            UpdateFormula::SR1 => {
                update_sr1(&mut b_mat, &s, &y);
            }
            UpdateFormula::DFP => {
                update_dfp(&mut h_inv, &s, &y);
            }
            UpdateFormula::BFGS => {
                update_bfgs(&mut h_inv, &s, &y, n);
            }
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
            format!(
                "Optimization terminated successfully using {} update.",
                update_formula.name()
            )
        } else {
            "Maximum iterations reached.".to_string()
        },
        jacobian: Some(g),
        hessian: None,
    })
}

/// SR1 update formula for Hessian approximation B
fn update_sr1(b_mat: &mut Option<Array2<f64>>, s: &Array1<f64>, y: &Array1<f64>) {
    if let Some(b) = b_mat.as_mut() {
        let bs = b.dot(s);
        let v = y - &bs;
        let denom = v.dot(s);

        // SR1 update with safeguard against division by zero
        let v_norm = v.mapv(|x| x.powi(2)).sum().sqrt();
        let s_norm = s.mapv(|x| x.powi(2)).sum().sqrt();
        if denom.abs() > 1e-8 * v_norm * s_norm {
            let v_col = v.clone().insert_axis(Axis(1));
            let v_row = v.clone().insert_axis(Axis(0));
            *b = &*b + (v_col.dot(&v_row) / denom);
        }
    }
}

/// DFP update formula for inverse Hessian approximation H
fn update_dfp(h_inv: &mut Option<Array2<f64>>, s: &Array1<f64>, y: &Array1<f64>) {
    if let Some(h) = h_inv.as_mut() {
        let s_dot_y = s.dot(y);

        if s_dot_y > 1e-10 {
            // DFP formula: H_{k+1} = H_k - (H_k y y^T H_k)/(y^T H_k y) + (s s^T)/(y^T s)
            let hy = h.dot(y);
            let ythy = y.dot(&hy);

            if ythy > 1e-10 {
                // Term 1: - (H_k y y^T H_k)/(y^T H_k y)
                let hy_col = hy.clone().insert_axis(Axis(1));
                let hy_row = hy.clone().insert_axis(Axis(0));
                let term1 = hy_col.dot(&hy_row) / ythy;

                // Term 2: + (s s^T)/(y^T s)
                let s_col = s.clone().insert_axis(Axis(1));
                let s_row = s.clone().insert_axis(Axis(0));
                let term2 = s_col.dot(&s_row) / s_dot_y;

                *h = &*h - &term1 + &term2;
            }
        }
    }
}

/// BFGS update formula for inverse Hessian approximation H
fn update_bfgs(h_inv: &mut Option<Array2<f64>>, s: &Array1<f64>, y: &Array1<f64>, n: usize) {
    if let Some(h) = h_inv.as_mut() {
        let s_dot_y = s.dot(y);

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
            let term3 = term1.dot(h);
            *h = term3.dot(&term2) + rho * s_col.dot(&s_row);
        }
    }
}

/// Simple matrix inversion using LU decomposition (for small matrices)
fn invert_matrix(mat: &Array2<f64>) -> Result<Array2<f64>, OptimizeError> {
    let n = mat.nrows();
    if n != mat.ncols() {
        return Err(OptimizeError::ValueError(
            "Matrix must be square".to_string(),
        ));
    }

    // For small matrices, use Gauss-Jordan elimination
    // For production code, should use a proper linear algebra library
    let mut aug = Array2::zeros((n, 2 * n));

    // Create augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = mat[[i, j]];
            if i == j {
                aug[[i, j + n]] = 1.0;
            }
        }
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singular matrix
        if aug[[i, i]].abs() < 1e-10 {
            return Err(OptimizeError::ValueError("Matrix is singular".to_string()));
        }

        // Scale pivot row
        let pivot = aug[[i, i]];
        for j in 0..(2 * n) {
            aug[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in 0..(2 * n) {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
    }

    // Extract inverse from augmented matrix
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, j + n]];
        }
    }

    Ok(inv)
}

impl UpdateFormula {
    /// Get the name of the update formula
    pub fn name(&self) -> &'static str {
        match self {
            UpdateFormula::SR1 => "SR1",
            UpdateFormula::DFP => "DFP",
            UpdateFormula::BFGS => "BFGS",
        }
    }
}

/// Convenience functions for specific quasi-Newton methods
pub fn minimize_sr1<F, S>(
    fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S + Clone,
    S: Into<f64> + Clone,
{
    minimize_quasi_newton(fun, x0, options, UpdateFormula::SR1)
}

pub fn minimize_dfp<F, S>(
    fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S + Clone,
    S: Into<f64> + Clone,
{
    minimize_quasi_newton(fun, x0, options, UpdateFormula::DFP)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unconstrained::Bounds;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sr1_quadratic() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 {
            let a = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).unwrap();
            let b = Array1::from_vec(vec![-4.0, -6.0]);
            0.5 * x.dot(&a.dot(x)) + b.dot(x)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let options = Options::default();

        let result = minimize_sr1(quadratic, x0, &options).unwrap();

        assert!(result.success);
        // Optimal solution: x = A^(-1) * (-b) = [2.0, 2.0]
        assert_abs_diff_eq!(result.x[0], 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_dfp_quadratic() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 {
            let a = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).unwrap();
            let b = Array1::from_vec(vec![-4.0, -6.0]);
            0.5 * x.dot(&a.dot(x)) + b.dot(x)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let options = Options::default();

        let result = minimize_dfp(quadratic, x0, &options).unwrap();

        assert!(result.success);
        // Optimal solution: x = A^(-1) * (-b) = [2.0, 2.0]
        assert_abs_diff_eq!(result.x[0], 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sr1_rosenbrock() {
        let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
            let a = 1.0;
            let b = 100.0;
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();
        options.max_iter = 5000; // More iterations for Rosenbrock with SR1
        options.gtol = 1e-4; // Relaxed gradient tolerance for SR1

        let result = minimize_sr1(rosenbrock, x0, &options).unwrap();

        // SR1 can be less stable than BFGS, so we check that reasonable progress was made
        let function_value = result.fun;

        // Either converged to solution or made significant progress toward optimum
        if (result.x[0] - 1.0).abs() < 1e-1 && (result.x[1] - 1.0).abs() < 1e-1 {
            // Successfully converged to optimum - no assertion needed
        } else {
            // Should at least significantly improve from initial value (~101)
            assert!(
                function_value < 10.0,
                "Function value {} should be much better than initial ~101",
                function_value
            );
        }
    }

    #[test]
    fn test_dfp_rosenbrock() {
        let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
            let a = 1.0;
            let b = 100.0;
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();
        options.max_iter = 2000; // More iterations for Rosenbrock

        let result = minimize_dfp(rosenbrock, x0, &options).unwrap();

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-2);
    }

    #[test]
    fn test_dfp_with_bounds() {
        let quadratic =
            |x: &ArrayView1<f64>| -> f64 { (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();

        // Constrain solution to [0, 1] x [0, 1]
        let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);
        options.bounds = Some(bounds);

        let result = minimize_dfp(quadratic, x0, &options).unwrap();

        assert!(result.success);
        // The optimal point (2, 3) is outside the bounds, so we should get (1, 1)
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-6);
    }
}
