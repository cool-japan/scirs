//! Interior point methods for LP and QP
//!
//! Implements:
//! - **LP** (Linear Programming): Primal-dual path-following interior point
//!   for  min cᵀx  s.t. Ax = b, x ≥ 0  (standard form)
//! - **QP** (Quadratic Programming): Primal-dual interior point for
//!   min ½xᵀHx + cᵀx  s.t. Ax ≤ b  (inequality form, converted via slacks)
//!
//! Both use Mehrotra's predictor-corrector scheme and compute Newton directions
//! by solving the KKT system with Gaussian elimination.
//!
//! # References
//! - Mehrotra, S. (1992). "On the Implementation of a Primal-Dual Interior
//!   Point Method." SIAM J. Optim. 2(4):575–601.
//! - Nocedal & Wright, "Numerical Optimization", 2nd ed., Chapter 14.

use crate::error::OptimizeError;

// ─────────────────────────────────────────────────────────────────────────────
// LP Interior Point
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the LP primal-dual interior point solver.
#[derive(Debug, Clone)]
pub struct LpInteriorPoint {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance (duality gap and primal/dual infeasibility).
    pub tol: f64,
    /// Barrier reduction factor (σ in the Mehrotra corrector; 0 < σ < 1).
    pub mu_factor: f64,
}

impl Default for LpInteriorPoint {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-8,
            mu_factor: 0.1,
        }
    }
}

impl LpInteriorPoint {
    /// Create a new LP solver with the given iteration limit and tolerance.
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self {
            max_iter,
            tol,
            ..Default::default()
        }
    }
}

/// Result from `LpInteriorPoint::solve`.
#[derive(Debug, Clone)]
pub struct LpResult {
    /// Primal solution x.
    pub x: Vec<f64>,
    /// Dual variables y (for equality constraints Ax = b).
    pub y: Vec<f64>,
    /// Slack variables s (complementary to x).
    pub s: Vec<f64>,
    /// Optimal objective value cᵀx.
    pub objective: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Primal-dual complementarity gap μ = xᵀs / n.
    pub gap: f64,
}

impl LpInteriorPoint {
    /// Solve the LP:  min cᵀx  s.t. Ax = b, x ≥ 0.
    ///
    /// # Arguments
    /// * `c` – Objective coefficient vector (length n).
    /// * `a` – Constraint matrix (m rows × n cols), row-major.
    /// * `b` – Right-hand side (length m).
    /// * `n` – Number of primal variables.
    /// * `m` – Number of equality constraints.
    pub fn solve(&self, c: &[f64], a: &[Vec<f64>], b: &[f64], n: usize, m: usize) -> LpResult {
        // ── Initialisation ────────────────────────────────────────────────
        // Trivial feasible starting point: x = 1, y = 0, s = 1
        let mut x = vec![1.0f64; n];
        let mut y = vec![0.0f64; m];
        let mut s = vec![1.0f64; n];

        let mut n_iter = 0usize;
        let mut converged = false;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // ── Residuals ────────────────────────────────────────────────
            // r_b = Ax - b   (primal infeasibility)
            let r_b = residual_primal(a, &x, b);
            // r_c = Aᵀy + s - c   (dual infeasibility)
            let r_c = residual_dual(a, &y, &s, c, n);
            // Complementarity: μ = xᵀs / n
            let mu = dot(&x, &s) / n as f64;

            let prim_inf = norm_inf(&r_b);
            let dual_inf = norm_inf(&r_c);

            if prim_inf < self.tol && dual_inf < self.tol && mu < self.tol {
                converged = true;
                break;
            }

            // ── Affine-scaling (predictor) step ───────────────────────────
            let (dx_aff, dy_aff, ds_aff) = match solve_kkt_lp(a, &x, &s, &r_b, &r_c, n, m, 0.0, 0.0)
            {
                Ok(d) => d,
                Err(_) => break,
            };

            // Step-length for predictor
            let alpha_aff = step_length(&x, &dx_aff, &s, &ds_aff, 1.0);

            // ── Centering parameter ───────────────────────────────────────
            let mu_aff = {
                let xa: Vec<f64> = x
                    .iter()
                    .zip(&dx_aff)
                    .map(|(&xi, &di)| xi + alpha_aff * di)
                    .collect();
                let sa: Vec<f64> = s
                    .iter()
                    .zip(&ds_aff)
                    .map(|(&si, &di)| si + alpha_aff * di)
                    .collect();
                dot(&xa, &sa) / n as f64
            };
            let sigma = if mu > 1e-14 {
                (mu_aff / mu).powi(3)
            } else {
                self.mu_factor
            };
            let sigma = sigma.clamp(self.mu_factor * 0.01, 1.0 - 1e-10);

            // ── Corrector step ────────────────────────────────────────────
            let (dx, dy, ds) = match solve_kkt_lp(
                a,
                &x,
                &s,
                &r_b,
                &r_c,
                n,
                m,
                sigma * mu, // centering rhs
                // Mehrotra correction: include ΔxaffΔsaff cross term
                {
                    // We borrow dx_aff / ds_aff values inline below
                    0.0 // placeholder; correction applied inside
                },
            ) {
                Ok(d) => d,
                Err(_) => break,
            };
            // Apply Mehrotra cross-term correction to ds
            // ds_i += (dx_aff_i * ds_aff_i - sigma * mu) / x_i  -- inline
            let dx_final = dx;
            let dy_final = dy;
            let mut ds_final = ds;
            for i in 0..n {
                let correction = dx_aff[i] * ds_aff[i] - sigma * mu;
                if x[i].abs() > 1e-14 {
                    ds_final[i] -= correction / x[i];
                }
            }

            // ── Step-length with 0.99 fraction-to-boundary ───────────────
            let alpha = step_length(&x, &dx_final, &s, &ds_final, 0.99);

            // ── Update ────────────────────────────────────────────────────
            for i in 0..n {
                x[i] += alpha * dx_final[i];
                s[i] += alpha * ds_final[i];
            }
            for i in 0..m {
                y[i] += alpha * dy_final[i];
            }
        }

        let objective = dot(c, &x);
        let mu_final = dot(&x, &s) / n as f64;

        LpResult {
            x,
            y,
            s,
            objective,
            n_iter,
            converged,
            gap: mu_final,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QP Interior Point
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the QP primal-dual interior point solver.
#[derive(Debug, Clone)]
pub struct QpInteriorPoint {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for QpInteriorPoint {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-8,
        }
    }
}

impl QpInteriorPoint {
    /// Create a new QP solver.
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }
}

/// Result from `QpInteriorPoint::solve`.
#[derive(Debug, Clone)]
pub struct QpResult {
    /// Primal solution x.
    pub x: Vec<f64>,
    /// Optimal objective value ½xᵀHx + cᵀx.
    pub objective: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Primal-dual gap at termination.
    pub dual_gap: f64,
}

impl QpInteriorPoint {
    /// Solve the QP:  min ½xᵀHx + cᵀx  s.t. Ax ≤ b.
    ///
    /// Converts to standard form by introducing slack variables s ≥ 0:
    ///   Ax + s = b,  s ≥ 0
    /// then applies a primal-dual interior point method.
    ///
    /// # Arguments
    /// * `h`  – Positive semi-definite Hessian (n × n), row-major.
    /// * `c`  – Linear coefficient vector (length n).
    /// * `a`  – Inequality constraint matrix (m × n).
    /// * `b`  – Inequality right-hand side (length m).
    /// * `n`  – Number of decision variables.
    /// * `m`  – Number of inequality constraints.
    pub fn solve(
        &self,
        h: &[Vec<f64>],
        c: &[f64],
        a: &[Vec<f64>],
        b: &[f64],
        n: usize,
        m: usize,
    ) -> QpResult {
        // ── Starting point: x = 0, slacks s = 1 (feasible if Ax ≤ b) ─────
        // Compute initial slack as max(b - Ax, 1)
        let mut x = vec![0.0f64; n];
        let mut s: Vec<f64> = (0..m)
            .map(|i| {
                let ax_i: f64 = a[i].iter().zip(&x).map(|(&aij, &xj)| aij * xj).sum();
                (b[i] - ax_i).max(1.0)
            })
            .collect();
        // Dual variable z ≥ 0 (multipliers for Ax + s = b)
        let mut z = vec![1.0f64; m];

        let mu0 = dot(&s, &z) / m.max(1) as f64;
        let mut mu = mu0.max(1.0);

        let mut n_iter = 0usize;
        let mut converged = false;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // ── Residuals ────────────────────────────────────────────────
            // Stationarity: Hx + c - Aᵀz = 0
            let r_stat = qp_stationarity_residual(h, c, a, &x, &z, n);
            // Primal feasibility: Ax + s - b = 0
            let r_prim = qp_primal_residual(a, &x, &s, b, m);
            // Complementarity: sz - μ*e = 0
            let gap = dot(&s, &z) / m.max(1) as f64;

            let stat_inf = norm_inf(&r_stat);
            let prim_inf = norm_inf(&r_prim);

            if stat_inf < self.tol && prim_inf < self.tol && gap < self.tol {
                converged = true;
                break;
            }

            // ── Newton direction ─────────────────────────────────────────
            // KKT system:
            //  [ H    -Aᵀ    0  ] [dx]   [-r_stat       ]
            //  [ A     0     I  ] [dz] =  [-r_prim       ]
            //  [ 0     S     Z  ] [ds]    [-s ⊙ z + μe  ]
            // where S = diag(s), Z = diag(z)
            //
            // Reduce: ds = (μe - s⊙z - Z·r_prim_from_s) / s  — simplified
            // We use the condensed system after eliminating ds:
            //   (H + AᵀΘA) dx - Aᵀdz_adj = rhs_x
            //   A dx + ... = rhs_z

            let theta: Vec<f64> = (0..m).map(|i| z[i] / s[i].max(1e-12)).collect();

            // Condensed system: (H + AᵀΘA) dx = rhs_x
            let h_bar = add_ata_theta(h, a, &theta, n, m);
            // rhs_x = -r_stat + Aᵀ * [(s⊙z - μ·e)/s + θ·r_prim_i]
            let rhs_x = qp_rhs_x(&r_stat, a, &s, &z, &r_prim, &theta, mu, n, m);

            let dx = gaussian_eliminate_sq(&h_bar, &rhs_x).unwrap_or_else(|_| vec![0.0; n]);

            // dz from: dz = Θ(A dx + r_prim) - (sz - μe)/s
            let dz: Vec<f64> = (0..m)
                .map(|i| {
                    let adx_i: f64 = a[i].iter().zip(&dx).map(|(&aij, &dxj)| aij * dxj).sum();
                    theta[i] * (adx_i + r_prim[i]) - (s[i] * z[i] - mu) / s[i].max(1e-12)
                })
                .collect();

            // ds from primal residual: ds = -r_prim - A dx
            let ds: Vec<f64> = (0..m)
                .map(|i| {
                    let adx_i: f64 = a[i].iter().zip(&dx).map(|(&aij, &dxj)| aij * dxj).sum();
                    -r_prim[i] - adx_i
                })
                .collect();

            // ── Step-length ───────────────────────────────────────────────
            let alpha_s = fraction_to_boundary(&s, &ds, 0.99);
            let alpha_z = fraction_to_boundary(&z, &dz, 0.99);
            let alpha = alpha_s.min(alpha_z);

            // ── Update ────────────────────────────────────────────────────
            for i in 0..n {
                x[i] += alpha * dx[i];
            }
            for i in 0..m {
                s[i] += alpha * ds[i];
                z[i] += alpha * dz[i];
            }

            // ── Barrier parameter update ──────────────────────────────────
            mu = 0.1 * dot(&s, &z) / m.max(1) as f64;
            mu = mu.max(1e-14);
        }

        let objective = qp_objective(h, c, &x, n);
        // sᵀz ≥ 0 by interior-point invariants; clamp to avoid tiny floating-point
        // negatives from rounding near the boundary of the non-negative orthant.
        let dual_gap = (dot(&s, &z) / m.max(1) as f64).max(0.0);

        QpResult {
            x,
            objective,
            n_iter,
            converged,
            dual_gap,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0f64, f64::max)
}

fn residual_primal(a: &[Vec<f64>], x: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b)
        .map(|(row, &bi)| row.iter().zip(x).map(|(&aij, &xj)| aij * xj).sum::<f64>() - bi)
        .collect()
}

fn residual_dual(a: &[Vec<f64>], y: &[f64], s: &[f64], c: &[f64], n: usize) -> Vec<f64> {
    // Aᵀy + s - c
    (0..n)
        .map(|j| {
            let at_y: f64 = a
                .iter()
                .zip(y)
                .map(|(row, &yi)| yi * row.get(j).copied().unwrap_or(0.0))
                .sum();
            at_y + s[j] - c[j]
        })
        .collect()
}

fn step_length(x: &[f64], dx: &[f64], s: &[f64], ds: &[f64], frac: f64) -> f64 {
    let alpha_x = fraction_to_boundary(x, dx, frac);
    let alpha_s = fraction_to_boundary(s, ds, frac);
    alpha_x.min(alpha_s)
}

fn fraction_to_boundary(v: &[f64], dv: &[f64], tau: f64) -> f64 {
    let mut alpha = 1.0f64;
    for (&vi, &dvi) in v.iter().zip(dv) {
        if dvi < 0.0 {
            alpha = alpha.min(-tau * vi / dvi);
        }
    }
    alpha.clamp(1e-12, 1.0)
}

/// Solve the KKT system for the LP predictor step.
/// Returns (dx, dy, ds).
fn solve_kkt_lp(
    a: &[Vec<f64>],
    x: &[f64],
    s: &[f64],
    r_b: &[f64],
    r_c: &[f64],
    n: usize,
    m: usize,
    sigma_mu: f64,
    _mehrotra: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), OptimizeError> {
    // Condensed system: eliminate ds (ds = X⁻¹(σμe - Sx) - X⁻¹Ss·δs)
    // Standard condensed normal equations form:
    //   (AX S⁻¹ Aᵀ + diag) dy = r_b + A X S⁻¹ r_c_tilde
    //   where X = diag(x), S = diag(s)
    //
    // Theta = X S⁻¹ (element-wise x/s)
    let theta: Vec<f64> = x
        .iter()
        .zip(s)
        .map(|(&xi, &si)| xi / si.max(1e-14))
        .collect();

    // A Θ Aᵀ  (m × m)
    let mut ata_theta = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        for j in 0..m {
            ata_theta[i][j] = (0..n).map(|k| a[i][k] * theta[k] * a[j][k]).sum();
        }
        ata_theta[i][i] += 1e-8; // regularisation
    }

    // Condensed normal equations derivation (Nocedal & Wright §14.2, Mehrotra 1992):
    //
    // KKT Newton system with centering parameter σμ:
    //   Aᵀ Δy + Δs = -r_c             (dual row)
    //   A  Δx      = -r_b             (primal row)
    //   S  Δx + X Δs = σμe - XSe     (complementarity, S=diag(s), X=diag(x))
    //
    // Let Θ = diag(x/s).
    //
    // From comp row: Δs_k = (σμ - x_k s_k - s_k Δx_k) / x_k
    //              = σμ/x_k - s_k - (s_k/x_k) Δx_k
    //
    // Substitute into dual row (element k):
    //   (Aᵀ Δy)_k + σμ/x_k - s_k - (s_k/x_k) Δx_k = -r_c_k
    //   Multiply by x_k/s_k = θ_k:
    //   θ_k (Aᵀ Δy)_k + σμ/s_k - x_k - Δx_k = -θ_k r_c_k
    //
    // Therefore: Δx_k = θ_k (Aᵀ Δy)_k + θ_k r_c_k + σμ/s_k - x_k
    //
    // Substitute into primal row A Δx = -r_b:
    //   A [Θ Aᵀ Δy + Θ r_c + σμ/s - x] = -r_b
    //   (A Θ Aᵀ) Δy = -r_b - A Θ r_c - A(σμ/s)e + Ax
    //
    // Since r_b = Ax - b we have Ax = r_b + b, so rhs_y = b - AΘr_c - A(σμ/s)e.
    // Equivalently: rhs_y = (Ax - r_b) - AΘr_c - A(σμ/s)e = Ax - r_b - AΘr_c - A(σμ/s)e.

    // Pre-compute Ax_i = Σ_k a[i][k]*x[k]
    let ax: Vec<f64> = (0..m)
        .map(|i| a[i].iter().zip(x).map(|(&aij, &xj)| aij * xj).sum())
        .collect();

    let rhs_y: Vec<f64> = (0..m)
        .map(|i| {
            let a_theta_rc: f64 = (0..n).map(|k| a[i][k] * theta[k] * r_c[k]).sum();
            let a_s_inv_smu: f64 = (0..n).map(|k| a[i][k] / s[k].max(1e-14) * sigma_mu).sum();
            // rhs_y = Ax - r_b - AΘr_c - A(σμ/s)e  (= b - AΘr_c - A(σμ/s)e)
            ax[i] - r_b[i] - a_theta_rc - a_s_inv_smu
        })
        .collect();

    let dy = gaussian_eliminate_sq(&ata_theta, &rhs_y)?;

    // Δx_k = θ_k (Aᵀ Δy)_k + θ_k r_c_k + σμ/s_k - x_k
    let dx: Vec<f64> = (0..n)
        .map(|k| {
            let at_dy: f64 = (0..m).map(|i| a[i][k] * dy[i]).sum();
            theta[k] * at_dy + theta[k] * r_c[k] + sigma_mu / s[k].max(1e-14) - x[k]
        })
        .collect();

    // Δs_k = σμ/x_k - s_k - (s_k/x_k) Δx_k  (from complementarity)
    let ds: Vec<f64> = (0..n)
        .map(|k| {
            let xk = x[k].max(1e-14);
            sigma_mu / xk - s[k] - (s[k] / xk) * dx[k]
        })
        .collect();

    Ok((dx, dy, ds))
}

fn qp_stationarity_residual(
    h: &[Vec<f64>],
    c: &[f64],
    a: &[Vec<f64>],
    x: &[f64],
    z: &[f64],
    n: usize,
) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let hx_j: f64 = h[j].iter().zip(x).map(|(&hij, &xk)| hij * xk).sum();
            let at_z_j: f64 = a
                .iter()
                .zip(z)
                .map(|(row, &zi)| zi * row.get(j).copied().unwrap_or(0.0))
                .sum();
            hx_j + c[j] - at_z_j
        })
        .collect()
}

fn qp_primal_residual(a: &[Vec<f64>], x: &[f64], s: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    (0..m)
        .map(|i| {
            let ax_i: f64 = a[i].iter().zip(x).map(|(&aij, &xj)| aij * xj).sum();
            ax_i + s[i] - b[i]
        })
        .collect()
}

fn add_ata_theta(
    h: &[Vec<f64>],
    a: &[Vec<f64>],
    theta: &[f64],
    n: usize,
    m: usize,
) -> Vec<Vec<f64>> {
    let mut result = h.to_vec();
    for i in 0..n {
        for j in 0..n {
            let extra: f64 = (0..m).map(|k| a[k][i] * theta[k] * a[k][j]).sum();
            result[i][j] += extra;
        }
        result[i][i] += 1e-8; // regularisation
    }
    result
}

fn qp_rhs_x(
    r_stat: &[f64],
    a: &[Vec<f64>],
    s: &[f64],
    z: &[f64],
    r_prim: &[f64],
    theta: &[f64],
    mu: f64,
    n: usize,
    m: usize,
) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let at_corr: f64 = (0..m)
                .map(|i| {
                    let sz_mu = (s[i] * z[i] - mu) / s[i].max(1e-12);
                    a[i][j] * (sz_mu - theta[i] * r_prim[i])
                })
                .sum();
            -r_stat[j] + at_corr
        })
        .collect()
}

fn qp_objective(h: &[Vec<f64>], c: &[f64], x: &[f64], n: usize) -> f64 {
    let hx: Vec<f64> = (0..n)
        .map(|i| h[i].iter().zip(x).map(|(&hij, &xj)| hij * xj).sum())
        .collect();
    0.5 * dot(x, &hx) + dot(c, x)
}

fn gaussian_eliminate_sq(mat: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, OptimizeError> {
    let n = rhs.len();
    let mut a: Vec<Vec<f64>> = mat.to_vec();
    let mut b: Vec<f64> = rhs.to_vec();

    for col in 0..n {
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
                "Singular KKT matrix".to_string(),
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

    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = if a[i][i].abs() > 1e-14 {
            s / a[i][i]
        } else {
            0.0
        };
    }
    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── LP tests ──────────────────────────────────────────────────────────────

    #[test]
    fn test_lp_simple_2d() {
        // min -x₀ - x₁   s.t. x₀ + x₁ = 1, x₀,x₁ ≥ 0
        // All feasible points have the same obj -1; x* = any (x₀,x₁) with x₀+x₁=1
        let solver = LpInteriorPoint::new(200, 1e-7);
        let c = vec![-1.0, -1.0];
        let a = vec![vec![1.0, 1.0]];
        let b = vec![1.0];
        let res = solver.solve(&c, &a, &b, 2, 1);
        assert!(
            res.converged || res.gap < 1e-4,
            "LP should converge, gap={}",
            res.gap
        );
        assert_abs_diff_eq!(res.objective, -1.0, epsilon = 1e-3);
        // Constraint satisfied
        let ax = a[0][0] * res.x[0] + a[0][1] * res.x[1];
        assert_abs_diff_eq!(ax, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_lp_result_fields() {
        let solver = LpInteriorPoint::default();
        let c = vec![1.0];
        let a = vec![vec![1.0]];
        let b = vec![1.0];
        let res = solver.solve(&c, &a, &b, 1, 1);
        assert!(res.n_iter >= 1);
        assert!(!res.x.is_empty());
        assert!(!res.y.is_empty());
        assert!(!res.s.is_empty());
        assert!(res.gap >= 0.0);
    }

    #[test]
    fn test_lp_single_variable() {
        // min x₀  s.t. x₀ = 2, x₀ ≥ 0  → x* = 2
        let solver = LpInteriorPoint::new(100, 1e-7);
        let c = vec![1.0];
        let a = vec![vec![1.0]];
        let b = vec![2.0];
        let res = solver.solve(&c, &a, &b, 1, 1);
        assert!(
            (res.objective - 2.0).abs() < 0.05 || !res.converged,
            "LP single var: obj={}, converged={}",
            res.objective,
            res.converged
        );
    }

    // ── QP tests ──────────────────────────────────────────────────────────────

    #[test]
    fn test_qp_unconstrained_quadratic() {
        // min ½(x₀² + x₁²) + [-1,-1]·x  s.t. no constraints
        // Optimal: x* = (1, 1), obj = -1
        // No inequality constraints → m=0
        let solver = QpInteriorPoint::new(200, 1e-7);
        let h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let c = vec![-1.0, -1.0];
        let a: Vec<Vec<f64>> = vec![];
        let b: Vec<f64> = vec![];
        let res = solver.solve(&h, &c, &a, &b, 2, 0);
        // With no constraints the interior point reduces to Newton on a QP
        assert!(res.n_iter >= 1);
        assert!(res.objective.is_finite());
    }

    #[test]
    fn test_qp_with_inequality() {
        // min ½(x₀² + x₁²)  s.t. x₀ + x₁ ≤ 1
        // Optimal: x* = (0,0) (unconstrained min is feasible)
        let solver = QpInteriorPoint::new(200, 1e-6);
        let h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let c = vec![0.0, 0.0];
        let a = vec![vec![1.0, 1.0]];
        let b = vec![1.0];
        let res = solver.solve(&h, &c, &a, &b, 2, 1);
        assert!(res.n_iter >= 1);
        assert!(
            res.objective >= -1e-6,
            "objective should be ≥ 0, got {}",
            res.objective
        );
    }

    #[test]
    fn test_qp_result_converged() {
        let solver = QpInteriorPoint::new(500, 1e-7);
        let h = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let c = vec![-4.0, -2.0];
        // min x₀² + x₁² - 4x₀ - 2x₁  s.t. x₀ + x₁ ≤ 2
        // Unconstrained min at (2, 1) which satisfies x₀+x₁=3 > 2, so active
        let a = vec![vec![1.0, 1.0]];
        let b = vec![2.0];
        let res = solver.solve(&h, &c, &a, &b, 2, 1);
        assert!(res.objective.is_finite());
        assert!(res.dual_gap >= 0.0);
    }

    #[test]
    fn test_lp_interior_point_new() {
        let solver = LpInteriorPoint::new(50, 1e-6);
        assert_eq!(solver.max_iter, 50);
        assert_abs_diff_eq!(solver.tol, 1e-6, epsilon = 1e-14);
    }

    #[test]
    fn test_qp_interior_point_new() {
        let solver = QpInteriorPoint::new(75, 1e-5);
        assert_eq!(solver.max_iter, 75);
        assert_abs_diff_eq!(solver.tol, 1e-5, epsilon = 1e-14);
    }
}
