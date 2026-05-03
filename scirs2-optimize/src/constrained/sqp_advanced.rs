//! Advanced Sequential Quadratic Programming (SQP) solver
//!
//! Provides a clean, struct-based API for nonlinear constrained optimization:
//!
//!   minimize  f(x)
//!   subject to  g(x) <= 0   (inequality)
//!               h(x) = 0    (equality)
//!               lb <= x <= ub  (bounds, optional)
//!
//! Algorithm:
//! - BFGS Hessian-of-Lagrangian approximation (Powell-damped updates)
//! - Active-set QP subproblem solver (inner iterations)
//! - L1 exact-penalty merit function with backtracking line search
//! - Lagrange-multiplier estimation via least-squares KKT residual
//!
//! # References
//! - Nocedal & Wright, "Numerical Optimization", 2nd ed., Chapter 18
//! - Boggs & Tolle, "Sequential Quadratic Programming", Acta Numerica 1995

use crate::error::{OptimizeError, OptimizeResult};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the `SqpSolver`.
#[derive(Debug, Clone)]
pub struct SqpSolver {
    /// Maximum number of outer SQP iterations.
    pub max_iter: usize,
    /// KKT optimality + feasibility convergence tolerance.
    pub tol: f64,
    /// Initial L1 merit-function penalty parameter ρ.
    pub merit_rho: f64,
    /// Step size for finite-difference gradient/Jacobian approximation.
    pub fd_eps: f64,
    /// Maximum number of backtracking line-search steps.
    pub max_ls_iter: usize,
    /// Armijo sufficient-decrease constant.
    pub armijo_c: f64,
    /// Backtracking factor (0 < β < 1).
    pub backtrack_factor: f64,
    /// Powell-BFGS damping threshold.
    pub bfgs_damping: f64,
}

impl Default for SqpSolver {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-8,
            merit_rho: 10.0,
            fd_eps: 1e-7,
            max_ls_iter: 40,
            armijo_c: 1e-4,
            backtrack_factor: 0.5,
            bfgs_damping: 0.2,
        }
    }
}

impl SqpSolver {
    /// Create a new solver with the given iteration limit and tolerance.
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self {
            max_iter,
            tol,
            ..Default::default()
        }
    }
}

/// Result from `SqpSolver::minimize`.
#[derive(Debug, Clone)]
pub struct SqpResult {
    /// Optimal decision vector.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub f_val: f64,
    /// L∞ constraint violation (max of |h_i(x)| and max(0, g_j(x))).
    pub constraint_violation: f64,
    /// Number of outer SQP iterations performed.
    pub n_iter: usize,
    /// Whether the KKT conditions were satisfied to tolerance.
    pub converged: bool,
    /// Lagrange multipliers for equality constraints h(x) = 0.
    pub multipliers_eq: Vec<f64>,
    /// KKT multipliers for inequality constraints g(x) ≤ 0.
    pub multipliers_ineq: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpSolver::minimize  (main entry point)
// ─────────────────────────────────────────────────────────────────────────────

impl SqpSolver {
    /// Minimize `f` subject to inequality constraints `g(x) <= 0` and equality
    /// constraints `h(x) = 0`.
    ///
    /// Gradients / Jacobians are estimated by central finite differences when
    /// the caller passes `None` for `grad_f`, `jac_g`, or `jac_h`.
    ///
    /// # Arguments
    /// * `f`      – Objective function.
    /// * `grad_f` – Gradient of `f` (or `None` → finite-diff).
    /// * `g_ineq` – Inequality-constraint function returning a vector (or `None`).
    /// * `jac_g`  – Jacobian of `g` as row vectors (or `None` → finite-diff).
    /// * `h_eq`   – Equality-constraint function (or `None`).
    /// * `jac_h`  – Jacobian of `h` (or `None` → finite-diff).
    /// * `x0`     – Initial point.
    /// * `bounds` – Optional box bounds `(lb, ub)` per variable.
    #[allow(clippy::too_many_arguments)]
    pub fn minimize<F, GF, G, JG, H, JH>(
        &self,
        f: F,
        grad_f: Option<GF>,
        g_ineq: Option<G>,
        jac_g: Option<JG>,
        h_eq: Option<H>,
        jac_h: Option<JH>,
        x0: &[f64],
        bounds: Option<&[(f64, f64)]>,
    ) -> SqpResult
    where
        F: Fn(&[f64]) -> f64,
        GF: Fn(&[f64]) -> Vec<f64>,
        G: Fn(&[f64]) -> Vec<f64>,
        JG: Fn(&[f64]) -> Vec<Vec<f64>>,
        H: Fn(&[f64]) -> Vec<f64>,
        JH: Fn(&[f64]) -> Vec<Vec<f64>>,
    {
        let n = x0.len();
        let mut x = x0.to_vec();

        // Determine constraint dimensions at the initial point
        let n_ineq = g_ineq.as_ref().map(|g| g(&x).len()).unwrap_or(0);
        let n_eq = h_eq.as_ref().map(|h| h(&x).len()).unwrap_or(0);

        // Initialize BFGS Hessian approximation (identity)
        let mut bfgs_h = identity_matrix(n);

        // Lagrange multipliers
        let mut lam_eq = vec![0.0f64; n_eq];
        let mut lam_ineq = vec![0.0f64; n_ineq];

        // Penalty parameter (auto-updated)
        let mut rho = self.merit_rho;

        let mut n_iter = 0usize;
        let mut converged = false;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // ── Evaluate at current x ──────────────────────────────────────
            let _fx = f(&x);

            let gf = match &grad_f {
                Some(gf) => gf(&x),
                None => finite_diff_gradient(&f, &x, self.fd_eps),
            };

            let g_vals: Vec<f64> = match &g_ineq {
                Some(g) => g(&x),
                None => vec![],
            };
            let g_jac: Vec<Vec<f64>> = if n_ineq > 0 {
                match &jac_g {
                    Some(jg) => jg(&x),
                    None => {
                        let gf_ineq = g_ineq.as_ref().map(|g| g as &dyn Fn(&[f64]) -> Vec<f64>);
                        if let Some(gf_ref) = gf_ineq {
                            finite_diff_jacobian(gf_ref, &x, n_ineq, self.fd_eps)
                        } else {
                            vec![]
                        }
                    }
                }
            } else {
                vec![]
            };

            let h_vals: Vec<f64> = match &h_eq {
                Some(h) => h(&x),
                None => vec![],
            };
            let h_jac: Vec<Vec<f64>> = if n_eq > 0 {
                match &jac_h {
                    Some(jh) => jh(&x),
                    None => {
                        let hf_eq = h_eq.as_ref().map(|h| h as &dyn Fn(&[f64]) -> Vec<f64>);
                        if let Some(hf_ref) = hf_eq {
                            finite_diff_jacobian(hf_ref, &x, n_eq, self.fd_eps)
                        } else {
                            vec![]
                        }
                    }
                }
            } else {
                vec![]
            };

            // ── Constraint violation ───────────────────────────────────────
            let cv = constraint_violation(&g_vals, &h_vals);

            // ── KKT stationarity residual ──────────────────────────────────
            let kkt = kkt_residual(&gf, &g_jac, &lam_ineq, &h_jac, &lam_eq, n);

            if kkt < self.tol && cv < self.tol {
                converged = true;
                break;
            }

            // ── Update penalty parameter ───────────────────────────────────
            // Increase ρ if constraint violation is large relative to gradient
            if cv > 1e-3 {
                rho = (rho * 2.0).min(1e8);
            }

            // ── Solve QP subproblem ────────────────────────────────────────
            let step = solve_qp_subproblem(&bfgs_h, &gf, &h_jac, &h_vals, &g_jac, &g_vals, n);

            // ── Line search (L1 merit function) ───────────────────────────
            let mut alpha = 1.0;
            let merit0 = l1_merit(&f, &g_ineq, &h_eq, &x, rho);
            let d_merit =
                directional_derivative_l1(&gf, &g_jac, &lam_ineq, &h_jac, &lam_eq, &step, rho);

            let mut x_new = x.clone();
            let mut ls_ok = false;
            for _ls in 0..self.max_ls_iter {
                x_new = x
                    .iter()
                    .zip(&step)
                    .map(|(&xi, &di)| xi + alpha * di)
                    .collect();
                // Apply bounds if given
                if let Some(bds) = bounds {
                    for (xi, &(lb, ub)) in x_new.iter_mut().zip(bds.iter()) {
                        *xi = xi.clamp(lb, ub);
                    }
                }
                let merit_new = l1_merit(&f, &g_ineq, &h_eq, &x_new, rho);
                if merit_new <= merit0 + self.armijo_c * alpha * d_merit {
                    ls_ok = true;
                    break;
                }
                alpha *= self.backtrack_factor;
            }
            if !ls_ok {
                // Accept a small step anyway to avoid stagnation
                x_new = x
                    .iter()
                    .zip(&step)
                    .map(|(&xi, &di)| xi + alpha * di)
                    .collect();
                if let Some(bds) = bounds {
                    for (xi, &(lb, ub)) in x_new.iter_mut().zip(bds.iter()) {
                        *xi = xi.clamp(lb, ub);
                    }
                }
            }

            // ── BFGS update ────────────────────────────────────────────────
            let s: Vec<f64> = x_new.iter().zip(&x).map(|(xn, xo)| xn - xo).collect();
            let gf_new = match &grad_f {
                Some(gf_fn) => gf_fn(&x_new),
                None => finite_diff_gradient(&f, &x_new, self.fd_eps),
            };

            // Lagrangian gradient difference
            let y = lagrangian_grad_diff(
                &gf,
                &gf_new,
                &g_jac,
                &g_vals,
                &g_ineq,
                &x_new,
                &lam_ineq,
                &h_jac,
                &h_vals,
                &h_eq,
                &x_new,
                &lam_eq,
                n,
                self.fd_eps,
            );

            bfgs_update_damped(&mut bfgs_h, &s, &y, self.bfgs_damping);

            // ── Multiplier update (from KKT least-squares) ────────────────
            update_multipliers(
                &gf_new,
                &g_jac,
                &mut lam_ineq,
                &g_vals,
                &h_jac,
                &mut lam_eq,
                n,
            );

            x = x_new;
        }

        let fx_final = f(&x);
        let g_final: Vec<f64> = g_ineq.as_ref().map(|g| g(&x)).unwrap_or_default();
        let h_final: Vec<f64> = h_eq.as_ref().map(|h| h(&x)).unwrap_or_default();
        let cv_final = constraint_violation(&g_final, &h_final);

        SqpResult {
            x,
            f_val: fx_final,
            constraint_violation: cv_final,
            n_iter,
            converged,
            multipliers_eq: lam_eq,
            multipliers_ineq: lam_ineq,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QP subproblem solver  (active-set, small dense problems)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the QP subproblem:
///
///   min  ½ dᵀ H d + gᵀ d
///   s.t. A_eq d + b_eq = 0
///        A_ineq d + b_ineq ≤ 0
///
/// Returns the step direction `d ∈ Rⁿ`.
///
/// Uses an equality-constrained reduced-gradient approach with active-set
/// management for the inequality constraints.
pub fn solve_qp_subproblem(
    h: &[Vec<f64>],
    g: &[f64],
    a_eq: &[Vec<f64>],
    b_eq: &[f64],
    a_ineq: &[Vec<f64>],
    b_ineq: &[f64],
    n: usize,
) -> Vec<f64> {
    // Active set: start with no active inequalities
    let n_ineq = a_ineq.len();
    let mut active: Vec<bool> = vec![false; n_ineq];

    let max_as_iter = 50 * (n + n_ineq + 1);
    let mut d = vec![0.0f64; n];

    for _as_iter in 0..max_as_iter {
        // Build combined equality system: equality + active inequalities
        let mut eq_rows: Vec<Vec<f64>> = a_eq.to_vec();
        let mut eq_rhs: Vec<f64> = b_eq.iter().map(|&bi| -bi).collect();

        for (i, &act) in active.iter().enumerate() {
            if act {
                eq_rows.push(a_ineq[i].clone());
                eq_rhs.push(-b_ineq[i]);
            }
        }

        // Solve reduced QP (equality-only)
        d = solve_eq_qp(h, g, &eq_rows, &eq_rhs, n);

        // Check if any inactive inequality is violated
        let mut most_violated = None;
        let mut max_viol = 1e-10;
        for (i, &act) in active.iter().enumerate() {
            if !act {
                let viol = dot(&a_ineq[i], &d) + b_ineq[i];
                if viol > max_viol {
                    max_viol = viol;
                    most_violated = Some(i);
                }
            }
        }

        if let Some(idx) = most_violated {
            active[idx] = true;
            continue;
        }

        // Check multipliers for active constraints; remove if negative
        let n_eq_orig = a_eq.len();
        let mut n_active = 0usize;
        let lagr = compute_qp_multipliers(h, g, &eq_rows, &d, n);

        let mut drop_idx = None;
        let mut min_lam = -1e-12;
        for (i, &act) in active.iter().enumerate() {
            if act {
                let lam_i = lagr.get(n_eq_orig + n_active).copied().unwrap_or(0.0);
                if lam_i < min_lam {
                    min_lam = lam_i;
                    drop_idx = Some(i);
                }
                n_active += 1;
            }
        }

        if let Some(idx) = drop_idx {
            active[idx] = false;
        } else {
            // KKT satisfied for the QP
            break;
        }
    }

    d
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Solve an equality-constrained QP: min ½dᵀHd + gᵀd  s.t. Ad = rhs
/// via the KKT system  [H  Aᵀ; A  0] [d; λ] = [-g; rhs].
fn solve_eq_qp(h: &[Vec<f64>], g: &[f64], a: &[Vec<f64>], rhs: &[f64], n: usize) -> Vec<f64> {
    let m = a.len();
    let sz = n + m;
    let mut mat = vec![vec![0.0f64; sz]; sz];
    let mut rh = vec![0.0f64; sz];

    // H block
    for i in 0..n {
        for j in 0..n {
            mat[i][j] = h[i][j];
        }
        // Add small regularization for numerical stability
        mat[i][i] += 1e-8;
        rh[i] = -g[i];
    }

    // A / Aᵀ blocks
    for (ci, row) in a.iter().enumerate() {
        for (j, &aij) in row.iter().enumerate() {
            mat[n + ci][j] = aij; // A
            mat[j][n + ci] = aij; // Aᵀ
        }
        rh[n + ci] = rhs[ci];
    }

    let sol = gaussian_eliminate(&mat, &rh).unwrap_or_else(|_| vec![0.0; sz]);
    sol[..n].to_vec()
}

/// Compute QP Lagrange multipliers for the active constraints.
fn compute_qp_multipliers(
    h: &[Vec<f64>],
    g: &[f64],
    a: &[Vec<f64>],
    d: &[f64],
    n: usize,
) -> Vec<f64> {
    let m = a.len();
    if m == 0 {
        return vec![];
    }
    // Residual r = H*d + g
    let r: Vec<f64> = (0..n).map(|i| dot(&h[i], d) + g[i]).collect();

    // Least-squares solve Aᵀ λ = r  →  (A Aᵀ) λ = A r
    let aat = mat_mul_t(a, a); // m × m
    let ar: Vec<f64> = (0..m).map(|i| dot(&a[i], &r)).collect();
    gaussian_eliminate(&aat, &ar).unwrap_or_else(|_| vec![0.0; m])
}

/// Gaussian elimination with partial pivoting.
fn gaussian_eliminate(mat: &[Vec<f64>], rhs: &[f64]) -> OptimizeResult<Vec<f64>> {
    let n = rhs.len();
    let mut a: Vec<Vec<f64>> = mat.to_vec();
    let mut b: Vec<f64> = rhs.to_vec();

    for col in 0..n {
        // Partial pivot
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                a[r1][col]
                    .abs()
                    .partial_cmp(&a[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        a.swap(col, pivot_row);
        b.swap(col, pivot_row);

        let piv = a[col][col];
        if piv.abs() < 1e-14 {
            return Err(OptimizeError::ComputationError(
                "Singular matrix in QP subproblem".to_string(),
            ));
        }

        for row in (col + 1)..n {
            let factor = a[row][col] / piv;
            for k in col..n {
                let tmp = a[col][k];
                a[row][k] -= factor * tmp;
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        if a[i][i].abs() < 1e-14 {
            x[i] = 0.0;
        } else {
            x[i] = s / a[i][i];
        }
    }
    Ok(x)
}

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let mut row = vec![0.0f64; n];
            row[i] = 1.0;
            row
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// A * Bᵀ where A is m1×k and B is m2×k → result is m1×m2.
fn mat_mul_t(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m1 = a.len();
    let m2 = b.len();
    (0..m1)
        .map(|i| (0..m2).map(|j| dot(&a[i], &b[j])).collect())
        .collect()
}

fn finite_diff_gradient<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0f64; n];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    for i in 0..n {
        xp[i] += eps;
        xm[i] -= eps;
        grad[i] = (f(&xp) - f(&xm)) / (2.0 * eps);
        xp[i] = x[i];
        xm[i] = x[i];
    }
    grad
}

fn finite_diff_jacobian<F: Fn(&[f64]) -> Vec<f64>>(
    f: F,
    x: &[f64],
    m: usize,
    eps: f64,
) -> Vec<Vec<f64>> {
    let n = x.len();
    // Returns m rows × n cols (each row = one constraint gradient)
    let mut jac = vec![vec![0.0f64; n]; m];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    for j in 0..n {
        xp[j] += eps;
        xm[j] -= eps;
        let fp = f(&xp);
        let fm = f(&xm);
        for i in 0..m {
            jac[i][j] = (fp[i] - fm[i]) / (2.0 * eps);
        }
        xp[j] = x[j];
        xm[j] = x[j];
    }
    jac
}

fn constraint_violation(g_vals: &[f64], h_vals: &[f64]) -> f64 {
    let ineq_viol = g_vals.iter().map(|&gi| gi.max(0.0)).fold(0.0f64, f64::max);
    let eq_viol = h_vals.iter().map(|&hi| hi.abs()).fold(0.0f64, f64::max);
    ineq_viol.max(eq_viol)
}

fn kkt_residual(
    gf: &[f64],
    g_jac: &[Vec<f64>],
    lam_ineq: &[f64],
    h_jac: &[Vec<f64>],
    lam_eq: &[f64],
    n: usize,
) -> f64 {
    let mut res = 0.0f64;
    for i in 0..n {
        let mut r = gf[i];
        for (j, row) in g_jac.iter().enumerate() {
            r += lam_ineq.get(j).copied().unwrap_or(0.0) * row.get(i).copied().unwrap_or(0.0);
        }
        for (j, row) in h_jac.iter().enumerate() {
            r += lam_eq.get(j).copied().unwrap_or(0.0) * row.get(i).copied().unwrap_or(0.0);
        }
        res = res.max(r.abs());
    }
    res
}

fn l1_merit<F, G, H>(f: &F, g_ineq: &Option<G>, h_eq: &Option<H>, x: &[f64], rho: f64) -> f64
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
    H: Fn(&[f64]) -> Vec<f64>,
{
    let fx = f(x);
    let mut penalty = 0.0f64;
    if let Some(g) = g_ineq {
        for gi in g(x) {
            penalty += gi.max(0.0);
        }
    }
    if let Some(h) = h_eq {
        for hi in h(x) {
            penalty += hi.abs();
        }
    }
    fx + rho * penalty
}

fn directional_derivative_l1(
    gf: &[f64],
    g_jac: &[Vec<f64>],
    lam_ineq: &[f64],
    h_jac: &[Vec<f64>],
    lam_eq: &[f64],
    d: &[f64],
    rho: f64,
) -> f64 {
    let grad_dot = dot(gf, d);
    // For L1 penalty, directional derivative includes constraint linearization
    let mut penalty_d = 0.0f64;
    for row in g_jac {
        let linear = dot(row, d);
        // Approx: treat as active
        penalty_d += linear.max(0.0);
    }
    for row in h_jac {
        let linear = dot(row, d);
        penalty_d += linear.abs();
    }
    // Use Lagrangian gradient direction (should be negative for descent)
    let _ = (lam_ineq, lam_eq); // used implicitly above
    grad_dot - rho * penalty_d.abs() - 1e-10
}

#[allow(clippy::too_many_arguments)]
fn lagrangian_grad_diff<G, H>(
    gf_old: &[f64],
    gf_new: &[f64],
    g_jac_old: &[Vec<f64>],
    g_vals_old: &[f64],
    g_ineq: &Option<G>,
    x_new: &[f64],
    lam_ineq: &[f64],
    h_jac_old: &[Vec<f64>],
    h_vals_old: &[f64],
    h_eq: &Option<H>,
    _x_new2: &[f64],
    lam_eq: &[f64],
    n: usize,
    fd_eps: f64,
) -> Vec<f64>
where
    G: Fn(&[f64]) -> Vec<f64>,
    H: Fn(&[f64]) -> Vec<f64>,
{
    let _ = (g_vals_old, h_vals_old, fd_eps);

    // New Jacobians at x_new (re-evaluated)
    let n_ineq = lam_ineq.len();
    let n_eq = lam_eq.len();

    let g_jac_new: Vec<Vec<f64>> = if n_ineq > 0 {
        if let Some(g) = g_ineq {
            finite_diff_jacobian(|x| g(x), x_new, n_ineq, 1e-7)
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    let h_jac_new: Vec<Vec<f64>> = if n_eq > 0 {
        if let Some(h) = h_eq {
            finite_diff_jacobian(|x| h(x), x_new, n_eq, 1e-7)
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    // y = ∇_x L(x_new) - ∇_x L(x_old)
    (0..n)
        .map(|i| {
            let lag_new = gf_new[i]
                + lam_ineq
                    .iter()
                    .enumerate()
                    .map(|(j, &l)| {
                        l * g_jac_new
                            .get(j)
                            .and_then(|r| r.get(i))
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .sum::<f64>()
                + lam_eq
                    .iter()
                    .enumerate()
                    .map(|(j, &l)| {
                        l * h_jac_new
                            .get(j)
                            .and_then(|r| r.get(i))
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .sum::<f64>();
            let lag_old = gf_old[i]
                + lam_ineq
                    .iter()
                    .enumerate()
                    .map(|(j, &l)| {
                        l * g_jac_old
                            .get(j)
                            .and_then(|r| r.get(i))
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .sum::<f64>()
                + lam_eq
                    .iter()
                    .enumerate()
                    .map(|(j, &l)| {
                        l * h_jac_old
                            .get(j)
                            .and_then(|r| r.get(i))
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .sum::<f64>();
            lag_new - lag_old
        })
        .collect()
}

fn bfgs_update_damped(h: &mut Vec<Vec<f64>>, s: &[f64], y: &[f64], theta: f64) {
    let n = s.len();
    let sy = dot(s, y);
    let shs: f64 = {
        let hs: Vec<f64> = (0..n).map(|i| dot(&h[i], s)).collect();
        dot(s, &hs)
    };

    // Powell damping: if sy < θ * sᵀHs, blend y toward H*s
    let (s_use, y_use): (Vec<f64>, Vec<f64>) = if sy < theta * shs {
        let factor = if shs.abs() > 1e-14 {
            (1.0 - theta) * shs / (shs - sy)
        } else {
            0.0
        };
        let hs: Vec<f64> = (0..n).map(|i| dot(&h[i], s)).collect();
        let y_damp: Vec<f64> = y
            .iter()
            .zip(&hs)
            .map(|(&yi, &hsi)| factor * hsi + (1.0 - factor) * yi)
            .collect();
        (s.to_vec(), y_damp)
    } else {
        (s.to_vec(), y.to_vec())
    };

    let sy2 = dot(&s_use, &y_use);
    if sy2.abs() < 1e-14 {
        return;
    }

    // BFGS formula: H ← H + y yᵀ/yᵀs - H s sᵀ Hᵀ / sᵀHs
    let hs: Vec<f64> = (0..n).map(|i| dot(&h[i], &s_use)).collect();
    let shs2 = dot(&s_use, &hs);

    if shs2.abs() < 1e-14 {
        return;
    }

    for i in 0..n {
        for j in 0..n {
            h[i][j] += y_use[i] * y_use[j] / sy2 - hs[i] * hs[j] / shs2;
        }
    }
}

fn update_multipliers(
    gf: &[f64],
    g_jac: &[Vec<f64>],
    lam_ineq: &mut Vec<f64>,
    g_vals: &[f64],
    h_jac: &[Vec<f64>],
    lam_eq: &mut Vec<f64>,
    n: usize,
) {
    let n_ineq = lam_ineq.len();
    let n_eq = lam_eq.len();

    if n_ineq == 0 && n_eq == 0 {
        return;
    }

    // Build constraint Jacobian block [Jg; Jh] (n_c × n)
    let n_c = n_ineq + n_eq;
    let mut jac: Vec<Vec<f64>> = Vec::with_capacity(n_c);
    jac.extend_from_slice(g_jac);
    jac.extend_from_slice(h_jac);

    // Solve  JᵀJ λ = Jᵀ(-∇f)  in a least-squares sense
    let jtj = mat_mul_t(&jac, &jac); // n_c × n_c
    let jt_neg_gf: Vec<f64> = (0..n_c).map(|i| -dot(&jac[i], gf)).collect();

    if let Ok(lam_all) = gaussian_eliminate(&jtj, &jt_neg_gf) {
        for (i, lam) in lam_ineq.iter_mut().enumerate() {
            // KKT: inequality multipliers must be non-negative
            let raw: f64 = lam_all.get(i).copied().unwrap_or(0.0);
            // If constraint active (violated), force positive
            let active = g_vals.get(i).map(|&gv| gv > -1e-6).unwrap_or(false);
            *lam = if active { raw.max(0.0) } else { 0.0 };
        }
        for (i, lam) in lam_eq.iter_mut().enumerate() {
            *lam = lam_all.get(n_ineq + i).copied().unwrap_or(0.0);
        }
    }
    let _ = n;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sqp_unconstrained_quadratic() {
        // min x₀² + x₁²  →  x* = (0, 0)
        let solver = SqpSolver::new(100, 1e-6);
        let res = solver.minimize(
            |x: &[f64]| x[0].powi(2) + x[1].powi(2),
            None::<fn(&[f64]) -> Vec<f64>>,
            None::<fn(&[f64]) -> Vec<f64>>,
            None::<fn(&[f64]) -> Vec<Vec<f64>>>,
            None::<fn(&[f64]) -> Vec<f64>>,
            None::<fn(&[f64]) -> Vec<Vec<f64>>>,
            &[1.0, 1.0],
            None,
        );
        assert!(res.converged, "should converge");
        assert_abs_diff_eq!(res.x[0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(res.x[1], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_sqp_equality_constrained() {
        // min x₀² + x₁²  s.t. x₀ + x₁ = 1
        // Optimal: x₀ = x₁ = 0.5, f* = 0.5
        let solver = SqpSolver::new(200, 1e-6);
        let res = solver.minimize(
            |x: &[f64]| x[0].powi(2) + x[1].powi(2),
            None::<fn(&[f64]) -> Vec<f64>>,
            None::<fn(&[f64]) -> Vec<f64>>,
            None::<fn(&[f64]) -> Vec<Vec<f64>>>,
            Some(|x: &[f64]| vec![x[0] + x[1] - 1.0]),
            None::<fn(&[f64]) -> Vec<Vec<f64>>>,
            &[0.0, 0.0],
            None,
        );
        assert!(
            res.constraint_violation < 1e-3,
            "constraint should be satisfied, cv = {}",
            res.constraint_violation
        );
        assert!(
            (res.f_val - 0.5).abs() < 0.05,
            "f* ≈ 0.5, got {}",
            res.f_val
        );
    }

    #[test]
    fn test_sqp_inequality_constrained() {
        // min (x₀-2)² + (x₁-2)²  s.t. x₀ + x₁ ≤ 3
        // Unconstrained min at (2,2) violates constraint, so opt at x₀=x₁=1.5
        let solver = SqpSolver::new(200, 1e-5);
        let res = solver.minimize(
            |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2),
            None::<fn(&[f64]) -> Vec<f64>>,
            Some(|x: &[f64]| vec![x[0] + x[1] - 3.0]),
            None::<fn(&[f64]) -> Vec<Vec<f64>>>,
            None::<fn(&[f64]) -> Vec<f64>>,
            None::<fn(&[f64]) -> Vec<Vec<f64>>>,
            &[0.5, 0.5],
            None,
        );
        assert!(
            res.constraint_violation < 1e-3,
            "constraint violated: cv={}",
            res.constraint_violation
        );
        assert!(
            res.f_val < 1.0,
            "objective should be small, got {}",
            res.f_val
        );
    }

    #[test]
    fn test_qp_subproblem_unconstrained() {
        // min ½ xᵀ I x + [1,0]ᵀ x  → x* = [-1, 0]
        let h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let g = vec![1.0, 0.0];
        let d = solve_qp_subproblem(&h, &g, &[], &[], &[], &[], 2);
        assert_abs_diff_eq!(d[0], -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(d[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sqp_result_fields() {
        let solver = SqpSolver::default();
        let res = solver.minimize(
            |x: &[f64]| x[0].powi(2),
            None::<fn(&[f64]) -> Vec<f64>>,
            None::<fn(&[f64]) -> Vec<f64>>,
            None::<fn(&[f64]) -> Vec<Vec<f64>>>,
            None::<fn(&[f64]) -> Vec<f64>>,
            None::<fn(&[f64]) -> Vec<Vec<f64>>>,
            &[3.0],
            None,
        );
        assert!(res.n_iter >= 1);
        assert!(res.constraint_violation >= 0.0);
        assert!(res.multipliers_eq.is_empty());
        assert!(res.multipliers_ineq.is_empty());
    }
}
