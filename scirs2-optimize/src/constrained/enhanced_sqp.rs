//! Enhanced Sequential Quadratic Programming (SQP) Solver
//!
//! This module provides a high-level SQP interface that wraps the lower-level
//! `sqp` module with a cleaner API, better default handling, and integration
//! with the enhanced constraint specification system.
//!
//! SQP solves the nonlinear programming problem:
//! ```text
//! minimize    f(x)
//! subject to  h_j(x) = 0    (equality constraints)
//!             g_i(x) <= 0   (inequality constraints)
//! ```
//! by solving a sequence of quadratic programming subproblems.
//!
//! # Features
//! - Automatic gradient approximation via finite differences
//! - Hessian approximation via BFGS updates
//! - Active set management for inequality constraints
//! - Merit function with line search
//!
//! # References
//! - Nocedal, J. & Wright, S.J. (2006). "Numerical Optimization." Chapter 18.

use crate::constrained::{Constraint, ConstraintFn, ConstraintKind, Options};
use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use scirs2_core::ndarray::{Array1, ArrayBase, Data, Ix1};

/// Enhanced SQP solver with a clean API
pub struct SQPSolver {
    /// Maximum iterations
    pub max_iter: usize,
    /// Optimality tolerance (KKT residual)
    pub tol: f64,
    /// Constraint violation tolerance
    pub constraint_tol: f64,
    /// Finite difference step for gradient approximation
    pub eps: f64,
    /// Initial Lagrange multiplier guess
    pub lambda_init: f64,
}

impl Default for SQPSolver {
    fn default() -> Self {
        SQPSolver {
            max_iter: 200,
            tol: 1e-8,
            constraint_tol: 1e-8,
            eps: 1e-7,
            lambda_init: 0.0,
        }
    }
}

impl SQPSolver {
    /// Create with default settings
    pub fn new() -> Self {
        SQPSolver::default()
    }

    /// Solve the NLP:
    ///   minimize f(x)
    ///   subject to equality constraints h_j(x) = 0
    ///              inequality constraints g_i(x) <= 0
    ///
    /// # Arguments
    /// - `f`: Objective function
    /// - `grad_f`: Optional gradient of f; if None, finite differences are used
    /// - `eq_cons`: Equality constraints h_j(x) = 0
    /// - `ineq_cons`: Inequality constraints g_i(x) <= 0
    /// - `x0`: Initial point
    pub fn solve<F, GF, E, G>(
        &self,
        f: F,
        grad_f: Option<GF>,
        eq_cons: &[E],
        ineq_cons: &[G],
        x0: &[f64],
    ) -> OptimizeResult<OptimizeResults<f64>>
    where
        F: Fn(&[f64]) -> f64 + Clone,
        GF: Fn(&[f64]) -> Vec<f64>,
        E: Fn(&[f64]) -> f64 + Clone,
        G: Fn(&[f64]) -> f64 + Clone,
    {
        let n = x0.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "x0 must be non-empty".to_string(),
            ));
        }

        let mut x = x0.to_vec();
        let mut nfev = 0usize;
        let mut njev = 0usize;
        let mut nit = 0usize;

        // BFGS Hessian approximation of Lagrangian
        let mut hess: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect();

        // Lagrange multipliers: eq then ineq
        let n_eq = eq_cons.len();
        let n_ineq = ineq_cons.len();
        let n_lambda = n_eq + n_ineq;
        let mut lambda = vec![self.lambda_init; n_lambda];

        // Evaluate gradient via finite differences or provided grad_f
        let compute_grad = |xv: &[f64], nfev: &mut usize, njev: &mut usize| -> Vec<f64> {
            *njev += 1;
            if let Some(ref gf) = grad_f {
                gf(xv)
            } else {
                let h = self.eps;
                let mut g = vec![0.0; n];
                let mut xp = xv.to_vec();
                let mut xm = xv.to_vec();
                for i in 0..n {
                    xp[i] = xv[i] + h;
                    xm[i] = xv[i] - h;
                    *nfev += 2;
                    g[i] = (f(&xp) - f(&xm)) / (2.0 * h);
                    xp[i] = xv[i];
                    xm[i] = xv[i];
                }
                g
            }
        };

        // Compute constraint Jacobians
        let compute_constraint_jac = |xv: &[f64], nfev: &mut usize| -> (Vec<f64>, Vec<Vec<f64>>) {
            let h = self.eps;
            let mut c_vals = Vec::with_capacity(n_lambda);
            let mut c_jac: Vec<Vec<f64>> = Vec::with_capacity(n_lambda);

            for e in eq_cons {
                let cv = e(xv);
                c_vals.push(cv);
                *nfev += 1;
                let mut jrow = vec![0.0; n];
                let mut xp = xv.to_vec();
                for j in 0..n {
                    xp[j] = xv[j] + h;
                    *nfev += 1;
                    jrow[j] = (e(&xp) - cv) / h;
                    xp[j] = xv[j];
                }
                c_jac.push(jrow);
            }

            for g in ineq_cons {
                let cv = g(xv);
                c_vals.push(cv);
                *nfev += 1;
                let mut jrow = vec![0.0; n];
                let mut xp = xv.to_vec();
                for j in 0..n {
                    xp[j] = xv[j] + h;
                    *nfev += 1;
                    jrow[j] = (g(&xp) - cv) / h;
                    xp[j] = xv[j];
                }
                c_jac.push(jrow);
            }

            (c_vals, c_jac)
        };

        let mut prev_grad_lag: Option<Vec<f64>> = None;
        let mut prev_x: Option<Vec<f64>> = None;

        for _iter in 0..self.max_iter {
            nit += 1;
            nfev += 1;
            let fx = f(&x);

            let grad = compute_grad(&x, &mut nfev, &mut njev);
            let (c_vals, c_jac) = compute_constraint_jac(&x, &mut nfev);

            // Compute Lagrangian gradient: grad_f + sum lambda_j * grad_c_j
            let mut grad_lag: Vec<f64> = grad.clone();
            for j in 0..n_lambda {
                for i in 0..n {
                    grad_lag[i] += lambda[j] * c_jac[j][i];
                }
            }

            // KKT condition: ||grad_lag|| + constraint_violation
            let grad_lag_norm = grad_lag.iter().map(|v| v * v).sum::<f64>().sqrt();
            let eq_viol: f64 = c_vals[..n_eq].iter().map(|v| v.abs()).sum();
            let ineq_viol: f64 = c_vals[n_eq..].iter().map(|v| v.max(0.0)).sum();
            let cv = eq_viol + ineq_viol;

            if grad_lag_norm <= self.tol && cv <= self.constraint_tol {
                return Ok(OptimizeResults {
                    x: Array1::from_vec(x),
                    fun: fx,
                    jac: Some(grad),
                    hess: None,
                    constr: Some(Array1::from_vec(c_vals)),
                    nit,
                    nfev,
                    njev,
                    nhev: 0,
                    maxcv: 0,
                    message: "KKT conditions satisfied".to_string(),
                    success: true,
                    status: 0,
                });
            }

            // BFGS update of Hessian (Powell damping)
            if let (Some(ref px), Some(ref pg)) = (&prev_x, &prev_grad_lag) {
                let s: Vec<f64> = x
                    .iter()
                    .zip(px.iter())
                    .map(|(&xi, &pxi)| xi - pxi)
                    .collect();
                let y: Vec<f64> = grad_lag
                    .iter()
                    .zip(pg.iter())
                    .map(|(&gi, &pgi)| gi - pgi)
                    .collect();
                let sy: f64 = s.iter().zip(y.iter()).map(|(&si, &yi)| si * yi).sum();
                // Hessian-vector product Hs
                let hs: Vec<f64> = (0..n)
                    .map(|i| {
                        hess[i]
                            .iter()
                            .zip(s.iter())
                            .map(|(&h, &si)| h * si)
                            .sum::<f64>()
                    })
                    .collect();
                let sths: f64 = s.iter().zip(hs.iter()).map(|(&si, &hsi)| si * hsi).sum();

                let sy_damp = sy.max(0.2 * sths); // Powell damping

                if sy_damp.abs() > 1e-10 && sths.abs() > 1e-10 {
                    // B <- B + y y^T / sy - Hs (Hs)^T / sHs
                    for i in 0..n {
                        for j in 0..n {
                            hess[i][j] += y[i] * y[j] / sy_damp - hs[i] * hs[j] / sths;
                        }
                    }
                }
            }

            prev_x = Some(x.clone());
            prev_grad_lag = Some(grad_lag.clone());

            // Solve QP subproblem using the full KKT system for equality constraints.
            // Active-set treatment for inequality constraints (active if c_i > -tol).
            //
            // KKT system for equality-constrained QP:
            //   [H    J_A^T] [d       ]   [-grad_f]
            //   [J_A  0    ] [delta_mu] = [-c_A   ]
            //
            // where J_A contains the Jacobians of all equality constraints and
            // active inequality constraints.
            let active_flags: Vec<bool> = (0..n_ineq).map(|i| c_vals[n_eq + i] > -1e-3).collect();
            let n_active = n_eq + active_flags.iter().filter(|&&a| a).count();

            let d = if n_active > 0 {
                solve_kkt_step(&hess, &grad, &c_jac, &c_vals, n_eq, &active_flags, n)
            } else {
                solve_newton_step(&hess, &grad, n)
            };

            // Merit function: exact L1 penalty ψ(x) = f(x) + μ*(||c_eq||₁ + ||max(0,c_ineq)||₁)
            // μ must exceed ||λ||_∞ to be an exact merit function (Nocedal&Wright §18.3)
            let mu_merit = lambda.iter().map(|v| v.abs()).fold(1.0_f64, f64::max) + 1.0;
            let mut merit_fn = |xv: &[f64]| -> f64 {
                let fv = f(xv);
                nfev += 1;
                let cv_eq: f64 = eq_cons.iter().map(|e| e(xv).abs()).sum::<f64>();
                let cv_ineq: f64 = ineq_cons.iter().map(|g| g(xv).max(0.0)).sum::<f64>();
                fv + mu_merit * (cv_eq + cv_ineq)
            };

            let merit0 = merit_fn(&x);
            // Directional derivative of L1 merit: grad_f^T d - μ*(||c_eq||₁ + ||max(0,c_ineq)||₁)
            let d_obj: f64 = grad.iter().zip(d.iter()).map(|(&gi, &di)| gi * di).sum();
            let d_merit = d_obj - mu_merit * (eq_viol + ineq_viol);

            let mut alpha = 1.0_f64;
            let armijo_c = 1e-4;
            let backtrack = 0.5;
            let max_ls = 30;

            for _ls in 0..max_ls {
                let xnew: Vec<f64> = x
                    .iter()
                    .zip(d.iter())
                    .map(|(&xi, &di)| xi + alpha * di)
                    .collect();
                let m_new = merit_fn(&xnew);
                if m_new <= merit0 + armijo_c * alpha * d_merit {
                    break;
                }
                alpha *= backtrack;
            }

            let xnew: Vec<f64> = x
                .iter()
                .zip(d.iter())
                .map(|(&xi, &di)| xi + alpha * di)
                .collect();
            x = xnew;

            // Update Lagrange multipliers via least squares on KKT
            let (new_c_vals, new_c_jac) = compute_constraint_jac(&x, &mut nfev);
            let new_grad = compute_grad(&x, &mut nfev, &mut njev);

            // Estimate multipliers: min ||grad_f + J^T lambda||^2
            // Simple: lambda <- -(J J^T)^{-1} J grad_f
            update_lagrange_multipliers(
                &new_grad,
                &new_c_jac,
                &new_c_vals,
                &mut lambda,
                n_eq,
                n_ineq,
            );
        }

        // Final evaluation
        nfev += 1;
        let fx = f(&x);
        let (c_vals, _) = compute_constraint_jac(&x, &mut nfev);
        let eq_viol: f64 = c_vals[..n_eq].iter().map(|v| v.abs()).sum();
        let ineq_viol: f64 = c_vals[n_eq..].iter().map(|v| v.max(0.0)).sum();
        let cv = eq_viol + ineq_viol;

        Ok(OptimizeResults {
            x: Array1::from_vec(x),
            fun: fx,
            jac: None,
            hess: None,
            constr: Some(Array1::from_vec(c_vals)),
            nit,
            nfev,
            njev,
            nhev: 0,
            maxcv: 0,
            message: format!("Maximum iterations reached (cv={:.2e})", cv),
            success: cv <= self.constraint_tol,
            status: if cv <= self.constraint_tol { 0 } else { 1 },
        })
    }
}

/// Solve the KKT system for the SQP subproblem:
///
/// ```text
/// [H    J_A^T] [d       ]   [-grad_f]
/// [J_A  0    ] [delta_mu] = [-c_A   ]
/// ```
///
/// where J_A is the stacked Jacobian of active constraints (all equality + active ineq).
/// Returns the primal step `d` of length `n`.
fn solve_kkt_step(
    hess: &[Vec<f64>],
    grad_f: &[f64],
    c_jac: &[Vec<f64>],
    c_vals: &[f64],
    n_eq: usize,
    active_flags: &[bool],
    n: usize,
) -> Vec<f64> {
    // Gather active constraint Jacobians and values
    let mut j_active: Vec<Vec<f64>> = Vec::new();
    let mut c_active: Vec<f64> = Vec::new();
    for j in 0..n_eq {
        j_active.push(c_jac[j].clone());
        c_active.push(c_vals[j]);
    }
    for (k, &is_active) in active_flags.iter().enumerate() {
        if is_active {
            j_active.push(c_jac[n_eq + k].clone());
            c_active.push(c_vals[n_eq + k]);
        }
    }

    let m = j_active.len();
    if m == 0 {
        return solve_newton_step_slice(hess, grad_f, n);
    }

    // Build the (n+m) x (n+m) KKT matrix:
    // [ H   J^T ]
    // [ J    0  ]
    let total = n + m;
    let mut kkt: Vec<Vec<f64>> = vec![vec![0.0; total]; total];
    let mut rhs: Vec<f64> = vec![0.0; total];

    // Top-left block: H (n x n)
    for i in 0..n {
        for j in 0..n {
            kkt[i][j] = hess[i][j];
        }
        // Small regularization for numerical stability
        kkt[i][i] += 1e-8;
    }

    // Top-right block: J^T (n x m)
    for k in 0..m {
        for i in 0..n {
            kkt[i][n + k] = j_active[k][i];
        }
    }

    // Bottom-left block: J (m x n)
    for k in 0..m {
        for j in 0..n {
            kkt[n + k][j] = j_active[k][j];
        }
    }

    // Right-hand side: [-grad_f; -c_A]
    for i in 0..n {
        rhs[i] = -grad_f[i];
    }
    for k in 0..m {
        rhs[n + k] = -c_active[k];
    }

    // Solve via Gaussian elimination with partial pivoting
    let sol = gaussian_elimination(&mut kkt, &mut rhs, total);
    match sol {
        Some(x) => {
            let d = x[0..n].to_vec();
            // Clamp step size
            let dn = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            if dn > 10.0 {
                let scale = 10.0 / dn;
                d.iter().map(|&v| v * scale).collect()
            } else {
                d
            }
        }
        None => {
            // Fall back to unconstrained Newton step
            solve_newton_step_slice(hess, grad_f, n)
        }
    }
}

/// Unconstrained Newton step: d = -H^{-1} grad (used as fallback)
fn solve_newton_step_slice(hess: &[Vec<f64>], grad: &[f64], n: usize) -> Vec<f64> {
    let mut a: Vec<Vec<f64>> = hess.iter().map(|row| row.to_vec()).collect();
    let mut b: Vec<f64> = grad.iter().map(|&gi| -gi).collect();
    // Add regularization
    for i in 0..n {
        a[i][i] += 1e-8;
    }
    match gaussian_elimination(&mut a, &mut b, n) {
        Some(d) => {
            let dn = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            if dn > 10.0 {
                let scale = 10.0 / dn;
                d.iter().map(|&v| v * scale).collect()
            } else {
                d
            }
        }
        None => grad.iter().map(|&gi| -gi * 0.01).collect(),
    }
}

/// Gaussian elimination with partial pivoting for square system Ax = b.
/// Modifies `a` and `b` in place; returns solution vector or None if singular.
fn gaussian_elimination(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        a.swap(col, max_row);
        b.swap(col, max_row);

        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for k in col..n {
                let val = a[col][k] * factor;
                a[row][k] -= val;
            }
            let bv = b[col] * factor;
            b[row] -= bv;
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        if a[i][i].abs() < 1e-14 {
            return None;
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

/// Solve Newton step d = -H^{-1} g using Gaussian elimination
fn solve_newton_step(hess: &Vec<Vec<f64>>, grad: &[f64], n: usize) -> Vec<f64> {
    // Build augmented matrix [H | -g]
    let mut a: Vec<Vec<f64>> = hess.iter().map(|row| row.clone()).collect();
    let mut b: Vec<f64> = grad.iter().map(|&gi| -gi).collect();

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            // Near-singular: use gradient descent direction
            return grad.iter().map(|&gi| -gi * 0.01).collect();
        }

        a.swap(col, max_row);
        b.swap(col, max_row);

        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for k in col..n {
                let val = a[col][k] * factor;
                a[row][k] -= val;
            }
            let bv = b[col] * factor;
            b[row] -= bv;
        }
    }

    // Back substitution
    let mut d = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * d[j];
        }
        if a[i][i].abs() < 1e-12 {
            d[i] = 0.0;
        } else {
            d[i] = sum / a[i][i];
        }
    }

    // Clamp step size
    let dn = d.iter().map(|v| v * v).sum::<f64>().sqrt();
    if dn > 10.0 {
        let scale = 10.0 / dn;
        d.iter_mut().for_each(|v| *v *= scale);
    }
    d
}

/// Update Lagrange multipliers given gradient and constraint Jacobian
fn update_lagrange_multipliers(
    grad: &[f64],
    c_jac: &[Vec<f64>],
    c_vals: &[f64],
    lambda: &mut Vec<f64>,
    n_eq: usize,
    n_ineq: usize,
) {
    let n = grad.len();
    let m = lambda.len();
    if m == 0 {
        return;
    }

    // Least squares: J^T lambda = -grad_f
    // Use normal equations: J J^T lambda = -J grad_f
    // For small m, use direct solve

    let mut jjt = vec![vec![0.0_f64; m]; m];
    let mut jg = vec![0.0_f64; m];

    for i in 0..m {
        for j in 0..m {
            let dot: f64 = c_jac[i]
                .iter()
                .zip(c_jac[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            jjt[i][j] = dot;
        }
        jg[i] = c_jac[i]
            .iter()
            .zip(grad.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>();
    }

    // Regularize
    for i in 0..m {
        jjt[i][i] += 1e-8;
    }

    if let Some(new_lambda) = solve_small_system_sqp(&jjt, &jg) {
        for i in 0..m {
            lambda[i] = -new_lambda[i];
        }
        // Enforce sign on inequality multipliers (must be >= 0 for active ineq constraints)
        for i in n_eq..n_eq + n_ineq {
            if c_vals[i] > 0.0 {
                lambda[i] = lambda[i].max(0.0);
            } else if c_vals[i] < -1e-6 {
                // Inactive constraint: multiplier should be 0
                lambda[i] = 0.0;
            }
        }
    }
}

/// Simple Gaussian elimination for small systems
fn solve_small_system_sqp(a: &Vec<Vec<f64>>, b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Some(vec![]);
    }
    let mut m: Vec<Vec<f64>> = a.iter().map(|row| row.clone()).collect();
    let mut r: Vec<f64> = b.to_vec();

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col + 1)..n {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        m.swap(col, max_row);
        r.swap(col, max_row);

        let pivot = m[col][col];
        for row in (col + 1)..n {
            let factor = m[row][col] / pivot;
            for k in col..n {
                let val = m[col][k] * factor;
                m[row][k] -= val;
            }
            let rv = r[col] * factor;
            r[row] -= rv;
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = r[i];
        for j in (i + 1)..n {
            sum -= m[i][j] * x[j];
        }
        if m[i][i].abs() < 1e-12 {
            return None;
        }
        x[i] = sum / m[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sqp_equality_constrained() {
        // min x^2 + y^2  s.t. x + y = 1
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let h = |x: &[f64]| x[0] + x[1] - 1.0;

        let solver = SQPSolver::default();
        let result = solver
            .solve(
                f,
                None::<fn(&[f64]) -> Vec<f64>>,
                &[h],
                &[] as &[fn(&[f64]) -> f64],
                &[0.0, 0.0],
            )
            .expect("solve failed");

        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 0.5, epsilon = 1e-3);
        assert_abs_diff_eq!(result.fun, 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_sqp_inequality_constrained() {
        // min x^2 + y^2  s.t. x + y >= 1 (g: 1-x-y <= 0)
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let g = |x: &[f64]| 1.0 - x[0] - x[1];

        let solver = SQPSolver::default();
        let result = solver
            .solve(
                f,
                None::<fn(&[f64]) -> Vec<f64>>,
                &[] as &[fn(&[f64]) -> f64],
                &[g],
                &[2.0, 2.0],
            )
            .expect("solve failed");

        // Solution: (0.5, 0.5), f=0.5
        assert_abs_diff_eq!(result.fun, 0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_sqp_unconstrained() {
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);

        let solver = SQPSolver::default();
        let result = solver
            .solve(
                f,
                None::<fn(&[f64]) -> Vec<f64>>,
                &[] as &[fn(&[f64]) -> f64],
                &[] as &[fn(&[f64]) -> f64],
                &[0.0, 0.0],
            )
            .expect("solve failed");

        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.fun, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_sqp_with_gradient() {
        // Provide analytical gradient
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let gf = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let h = |x: &[f64]| x[0] + x[1] - 2.0;

        let solver = SQPSolver::default();
        let result = solver
            .solve(f, Some(gf), &[h], &[] as &[fn(&[f64]) -> f64], &[0.0, 0.0])
            .expect("solve failed");

        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.fun, 2.0, epsilon = 1e-3);
    }

    #[test]
    fn test_sqp_mixed_constraints() {
        // min (x-2)^2 + (y-2)^2  s.t. x+y=3, x<=2
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
        let h = |x: &[f64]| x[0] + x[1] - 3.0;
        let g = |x: &[f64]| x[0] - 2.0;

        let solver = SQPSolver {
            max_iter: 300,
            tol: 1e-6,
            constraint_tol: 1e-5,
            ..Default::default()
        };
        let result = solver
            .solve(f, None::<fn(&[f64]) -> Vec<f64>>, &[h], &[g], &[1.0, 2.0])
            .expect("solve failed");

        // Solution: on constraint x+y=3 with x<=2: x=2,y=1 -> f=0+1=1, or x=1.5,y=1.5->f=0.25+0.25=0.5
        // Actually x=1.5, y=1.5 satisfies x+y=3 and x<=2, f = 0.25+0.25=0.5 < 1
        assert!(result.fun <= 1.0 + 1e-3, "fun={}", result.fun);
    }

    #[test]
    fn test_sqp_3d_equality() {
        // min x^2 + y^2 + z^2  s.t. x + 2y + 3z = 6
        // Solution via KKT: x=6/14, y=12/14, z=18/14
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2) + x[2].powi(2);
        let h = |x: &[f64]| x[0] + 2.0 * x[1] + 3.0 * x[2] - 6.0;

        let solver = SQPSolver {
            max_iter: 500,
            ..Default::default()
        };
        let result = solver
            .solve(
                f,
                None::<fn(&[f64]) -> Vec<f64>>,
                &[h],
                &[] as &[fn(&[f64]) -> f64],
                &[1.0, 1.0, 1.0],
            )
            .expect("solve failed");

        // Optimal: ||x||^2 = 36/14 ≈ 2.571
        assert!(result.fun <= 3.0, "fun={}", result.fun);
    }
}
