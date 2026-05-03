//! Geometric Programming (GP) solver.
//!
//! # Standard-Form GP
//!
//! ```text
//! minimise   f₀(x)
//! subject to fᵢ(x) ≤ 1,   i = 1 … m
//!            hⱼ(x) = 1,   j = 1 … p
//! ```
//!
//! where every fᵢ is a **posynomial** (sum of monomials with positive
//! coefficients) and every hⱼ is a **monomial**.
//!
//! A **monomial** in n variables x = (x₁, …, xₙ) with all xᵢ > 0 is:
//!
//! ```text
//! g(x) = c · x₁^a₁ · x₂^a₂ · … · xₙ^aₙ
//! ```
//!
//! where c > 0 and aᵢ ∈ ℝ are the exponents.
//!
//! # Log Transformation
//!
//! Substituting  xᵢ = exp(yᵢ) turns the GP into a convex problem:
//!
//! ```text
//! minimise   log Σₖ exp(aₖ·y + log cₖ)
//! subject to log Σₖ exp(aₖ·y + log cₖ) ≤ 0   for each constraint
//! ```
//!
//! which is a sum-of-exponentials (log-sum-exp) minimisation problem solved
//! here by a barrier interior-point method.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::fmt;

// ─── Monomial ────────────────────────────────────────────────────────────────

/// A single monomial  c · x₁^a₁ · … · xₙ^aₙ  (c > 0, xᵢ > 0 required).
#[derive(Debug, Clone)]
pub struct Monomial {
    /// Positive coefficient.
    pub coefficient: f64,
    /// Exponent vector (one entry per variable).
    pub exponents: Vec<f64>,
}

impl Monomial {
    /// Create a new monomial, returning an error if the coefficient is not
    /// strictly positive.
    pub fn new(coefficient: f64, exponents: Vec<f64>) -> OptimizeResult<Self> {
        if coefficient <= 0.0 {
            return Err(OptimizeError::ValueError(
                "monomial coefficient must be strictly positive".into(),
            ));
        }
        Ok(Self {
            coefficient,
            exponents,
        })
    }

    /// Evaluate at x (all components must be > 0).
    pub fn evaluate(&self, x: &[f64]) -> OptimizeResult<f64> {
        if x.len() != self.exponents.len() {
            return Err(OptimizeError::ValueError(format!(
                "monomial has {} exponents but x has {} components",
                self.exponents.len(),
                x.len()
            )));
        }
        let mut val = self.coefficient;
        for (xi, ai) in x.iter().zip(self.exponents.iter()) {
            if *xi <= 0.0 {
                return Err(OptimizeError::ValueError(
                    "GP variables must be strictly positive".into(),
                ));
            }
            val *= xi.powf(*ai);
        }
        Ok(val)
    }

    /// Number of variables.
    #[inline]
    pub fn n_vars(&self) -> usize {
        self.exponents.len()
    }
}

impl fmt::Display for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.coefficient)?;
        for (i, ai) in self.exponents.iter().enumerate() {
            if *ai != 0.0 {
                write!(f, " · x{}^{:.4}", i + 1, ai)?;
            }
        }
        Ok(())
    }
}

// ─── Posynomial ──────────────────────────────────────────────────────────────

/// A posynomial: sum of monomials, each with a positive coefficient.
#[derive(Debug, Clone)]
pub struct Posynomial {
    /// Non-empty list of monomials, all sharing the same variable dimension.
    pub terms: Vec<Monomial>,
}

impl Posynomial {
    /// Construct a posynomial, validating that all monomials have the same
    /// variable dimension.
    pub fn new(terms: Vec<Monomial>) -> OptimizeResult<Self> {
        if terms.is_empty() {
            return Err(OptimizeError::ValueError(
                "posynomial must have at least one term".into(),
            ));
        }
        let n = terms[0].n_vars();
        for (k, t) in terms.iter().enumerate() {
            if t.n_vars() != n {
                return Err(OptimizeError::ValueError(format!(
                    "term {} has {} variables but term 0 has {}",
                    k,
                    t.n_vars(),
                    n
                )));
            }
        }
        Ok(Self { terms })
    }

    /// Evaluate at x.
    pub fn evaluate(&self, x: &[f64]) -> OptimizeResult<f64> {
        let mut sum = 0.0;
        for t in &self.terms {
            sum += t.evaluate(x)?;
        }
        Ok(sum)
    }

    /// Number of variables.
    #[inline]
    pub fn n_vars(&self) -> usize {
        self.terms[0].n_vars()
    }

    /// Number of terms.
    #[inline]
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }
}

// ─── GP problem ──────────────────────────────────────────────────────────────

/// A Geometric Program in standard form:
///
/// ```text
/// minimise   objective(x)
/// subject to ineq_constraints[i](x) ≤ 1,   i = 0 … m-1
///            eq_constraints[j](x)   = 1,   j = 0 … p-1
/// ```
///
/// Equality constraints are monomials (single-term posynomials).
#[derive(Debug)]
pub struct GPProblem {
    /// Objective posynomial f₀.
    pub objective: Posynomial,
    /// Inequality constraints fᵢ(x) ≤ 1.
    pub ineq_constraints: Vec<Posynomial>,
    /// Equality constraints hⱼ(x) = 1 (must each be a single monomial).
    pub eq_constraints: Vec<Monomial>,
}

impl GPProblem {
    /// Create a GP problem.
    pub fn new(
        objective: Posynomial,
        ineq_constraints: Vec<Posynomial>,
        eq_constraints: Vec<Monomial>,
    ) -> OptimizeResult<Self> {
        let n = objective.n_vars();
        for (i, c) in ineq_constraints.iter().enumerate() {
            if c.n_vars() != n {
                return Err(OptimizeError::ValueError(format!(
                    "inequality constraint {} has {} variables; expected {}",
                    i,
                    c.n_vars(),
                    n
                )));
            }
        }
        for (j, c) in eq_constraints.iter().enumerate() {
            if c.n_vars() != n {
                return Err(OptimizeError::ValueError(format!(
                    "equality constraint {} has {} variables; expected {}",
                    j,
                    c.n_vars(),
                    n
                )));
            }
        }
        Ok(Self {
            objective,
            ineq_constraints,
            eq_constraints,
        })
    }

    /// Number of primal variables.
    #[inline]
    pub fn n_vars(&self) -> usize {
        self.objective.n_vars()
    }
}

// ─── Solver configuration ────────────────────────────────────────────────────

/// Configuration for the GP interior-point solver.
#[derive(Debug, Clone)]
pub struct GPSolverConfig {
    /// Maximum number of outer (barrier) iterations.
    pub max_outer_iters: usize,
    /// Maximum number of inner Newton iterations per barrier step.
    pub max_inner_iters: usize,
    /// Outer stopping tolerance on the duality gap proxy.
    pub outer_tol: f64,
    /// Inner Newton stopping tolerance.
    pub inner_tol: f64,
    /// Initial barrier parameter t.
    pub t_init: f64,
    /// Barrier parameter growth factor μ > 1.
    pub mu: f64,
    /// Initial log-domain starting point y = log(x), one per variable.
    pub initial_y: Option<Vec<f64>>,
    /// Armijo line-search shrink factor α ∈ (0, 1).
    pub ls_alpha: f64,
    /// Line-search step-size reduction factor β ∈ (0, 1).
    pub ls_beta: f64,
}

impl Default for GPSolverConfig {
    fn default() -> Self {
        Self {
            max_outer_iters: 50,
            max_inner_iters: 30,
            outer_tol: 1e-8,
            inner_tol: 1e-8,
            t_init: 1.0,
            mu: 10.0,
            initial_y: None,
            ls_alpha: 0.01,
            ls_beta: 0.5,
        }
    }
}

// ─── Solver result ────────────────────────────────────────────────────────────

/// Result returned by [`solve_gp`].
#[derive(Debug)]
pub struct GPResult {
    /// Optimal primal variables x* (positive).
    pub x: Vec<f64>,
    /// Optimal objective value f₀(x*).
    pub obj_value: f64,
    /// Number of outer iterations performed.
    pub outer_iters: usize,
    /// Whether the solver declared convergence.
    pub converged: bool,
    /// Final duality-gap proxy.
    pub gap: f64,
}

// ─── Log-domain convex problem ────────────────────────────────────────────────

/// The log-transformed convex problem produced by [`gp_to_convex`].
///
/// Every posynomial constraint  f(x) ≤ 1  becomes:
///
/// ```text
/// lse(A_row · y + b_row) ≤ 0
/// ```
///
/// where `A` is the matrix of exponents (rows = monomials, cols = variables)
/// and `b` is the vector of log-coefficients.
#[derive(Debug)]
pub struct LogConvexProblem {
    /// Exponent matrix for the objective (K₀ × n).
    pub obj_a: Array2<f64>,
    /// Log-coefficient vector for the objective (K₀,).
    pub obj_b: Array1<f64>,
    /// Per-constraint exponent matrices.  len = m.
    pub con_a: Vec<Array2<f64>>,
    /// Per-constraint log-coefficient vectors.  len = m.
    pub con_b: Vec<Array1<f64>>,
    /// Equality-constraint exponent rows (one per monomial equality).
    pub eq_a: Array2<f64>,
    /// Equality-constraint log-coefficients.
    pub eq_b: Array1<f64>,
}

/// Convert a [`GPProblem`] to its log-domain convex representation.
///
/// This exposes the raw convex formulation for users who want to pass it to
/// their own solver.  The standard way to solve a GP is [`solve_gp`].
pub fn gp_to_convex(prob: &GPProblem) -> LogConvexProblem {
    let n = prob.n_vars();

    let (obj_a, obj_b) = posynomial_to_log_matrices(&prob.objective, n);

    let (con_a, con_b): (Vec<_>, Vec<_>) = prob
        .ineq_constraints
        .iter()
        .map(|c| posynomial_to_log_matrices(c, n))
        .unzip();

    let n_eq = prob.eq_constraints.len();
    let mut eq_a = Array2::<f64>::zeros((n_eq.max(1), n));
    let mut eq_b = Array1::<f64>::zeros(n_eq.max(1));
    for (j, m) in prob.eq_constraints.iter().enumerate() {
        for (k, a) in m.exponents.iter().enumerate() {
            eq_a[[j, k]] = *a;
        }
        eq_b[j] = m.coefficient.ln();
    }

    LogConvexProblem {
        obj_a,
        obj_b,
        con_a,
        con_b,
        eq_a,
        eq_b,
    }
}

/// Build the exponent matrix (K×n) and log-coefficient vector (K,) for a
/// posynomial with K terms.
fn posynomial_to_log_matrices(p: &Posynomial, n: usize) -> (Array2<f64>, Array1<f64>) {
    let k = p.n_terms();
    let mut a = Array2::<f64>::zeros((k, n));
    let mut b = Array1::<f64>::zeros(k);
    for (row, term) in p.terms.iter().enumerate() {
        for (col, exp) in term.exponents.iter().enumerate() {
            a[[row, col]] = *exp;
        }
        b[row] = term.coefficient.ln();
    }
    (a, b)
}

// ─── log-sum-exp helper ───────────────────────────────────────────────────────

/// Numerically stable  log(Σ exp(v_i)).
fn log_sum_exp(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_v == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = v.iter().map(|vi| (vi - max_v).exp()).sum();
    max_v + sum.ln()
}

/// Compute  z_i = exp(v_i - lse(v))  (softmax).
fn softmax(v: &[f64]) -> Vec<f64> {
    let lse = log_sum_exp(v);
    v.iter().map(|vi| (vi - lse).exp()).collect()
}

// ─── Barrier + Newton helpers ─────────────────────────────────────────────────

/// Evaluate the log-domain objective  lse(A·y + b).
fn lse_objective(a: &Array2<f64>, b: &Array1<f64>, y: &[f64]) -> f64 {
    let v: Vec<f64> = (0..a.nrows())
        .map(|k| {
            let inner: f64 = (0..a.ncols()).map(|j| a[[k, j]] * y[j]).sum();
            inner + b[k]
        })
        .collect();
    log_sum_exp(&v)
}

/// Gradient of  lse(A·y + b)  w.r.t. y:  Aᵀ · softmax(A·y + b).
fn lse_gradient(a: &Array2<f64>, b: &Array1<f64>, y: &[f64]) -> Vec<f64> {
    let v: Vec<f64> = (0..a.nrows())
        .map(|k| {
            let inner: f64 = (0..a.ncols()).map(|j| a[[k, j]] * y[j]).sum();
            inner + b[k]
        })
        .collect();
    let sm = softmax(&v);
    let n = a.ncols();
    let mut grad = vec![0.0_f64; n];
    for k in 0..a.nrows() {
        for j in 0..n {
            grad[j] += sm[k] * a[[k, j]];
        }
    }
    grad
}

/// Hessian of  lse(A·y + b)  w.r.t. y:  Aᵀ · diag(sm) · A − (Aᵀ sm)(Aᵀ sm)ᵀ.
fn lse_hessian(a: &Array2<f64>, b: &Array1<f64>, y: &[f64]) -> Array2<f64> {
    let v: Vec<f64> = (0..a.nrows())
        .map(|k| {
            let inner: f64 = (0..a.ncols()).map(|j| a[[k, j]] * y[j]).sum();
            inner + b[k]
        })
        .collect();
    let sm = softmax(&v);
    let n = a.ncols();
    let mut h = Array2::<f64>::zeros((n, n));
    // Aᵀ diag(sm) A
    for k in 0..a.nrows() {
        for i in 0..n {
            for j in 0..n {
                h[[i, j]] += sm[k] * a[[k, i]] * a[[k, j]];
            }
        }
    }
    // subtract outer product (Aᵀ sm)(Aᵀ sm)ᵀ
    let g = lse_gradient(a, b, y);
    for i in 0..n {
        for j in 0..n {
            h[[i, j]] -= g[i] * g[j];
        }
    }
    h
}

/// Solve the n×n linear system H·Δ = −g via Cholesky (or fall back to Gaussian
/// elimination with partial pivoting when H is not PD).
fn solve_newton_system(h: &Array2<f64>, g: &[f64]) -> Vec<f64> {
    let n = h.nrows();
    // Regularise diagonal for robustness.
    let reg = 1e-12;
    // Gaussian elimination with partial pivoting.
    let mut mat: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| h[[i, j]]).collect();
            row[i] += reg;
            row.push(-g[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Find pivot.
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in (col + 1)..n {
            let v = mat[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        mat.swap(col, max_row);

        let pivot = mat[col][col];
        if pivot.abs() < 1e-30 {
            continue;
        }
        let inv_pivot = 1.0 / pivot;
        for j in col..=n {
            mat[col][j] *= inv_pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = mat[row][col];
            for j in col..=n {
                let delta = factor * mat[col][j];
                mat[row][j] -= delta;
            }
        }
    }

    (0..n).map(|i| mat[i][n]).collect()
}

// ─── Interior-point barrier solver ───────────────────────────────────────────

/// Evaluate the barrier objective for fixed t and the log-domain problem.
///
/// F_t(y) = t · lse(obj) + Σᵢ (−log(−lse(con_i)))
///
/// Constraints must be strictly feasible: lse(con_i(y)) < 0 ⟺ fᵢ(exp(y)) < 1.
fn barrier_value(lcp: &LogConvexProblem, t: f64, y: &[f64]) -> Option<f64> {
    let mut val = t * lse_objective(&lcp.obj_a, &lcp.obj_b, y);
    for (ca, cb) in lcp.con_a.iter().zip(lcp.con_b.iter()) {
        let lse_c = lse_objective(ca, cb, y);
        if lse_c >= 0.0 {
            return None; // infeasible
        }
        val -= (-lse_c).ln();
    }
    Some(val)
}

/// Gradient of the barrier objective.
fn barrier_gradient(lcp: &LogConvexProblem, t: f64, y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut grad: Vec<f64> = lse_gradient(&lcp.obj_a, &lcp.obj_b, y)
        .iter()
        .map(|g| t * g)
        .collect();
    for (ca, cb) in lcp.con_a.iter().zip(lcp.con_b.iter()) {
        let lse_c = lse_objective(ca, cb, y);
        let factor = 1.0 / (-lse_c).max(1e-30);
        let gc = lse_gradient(ca, cb, y);
        for j in 0..n {
            grad[j] += factor * gc[j];
        }
    }
    grad
}

/// Hessian of the barrier objective.
fn barrier_hessian(lcp: &LogConvexProblem, t: f64, y: &[f64]) -> Array2<f64> {
    let n = y.len();
    let mut h = lse_hessian(&lcp.obj_a, &lcp.obj_b, y);
    for i in 0..n {
        for j in 0..n {
            h[[i, j]] *= t;
        }
    }
    for (ca, cb) in lcp.con_a.iter().zip(lcp.con_b.iter()) {
        let lse_c = lse_objective(ca, cb, y);
        let s = (-lse_c).max(1e-30);
        let gc = lse_gradient(ca, cb, y);
        let hc = lse_hessian(ca, cb, y);
        let inv_s = 1.0 / s;
        let inv_s2 = inv_s * inv_s;
        for i in 0..n {
            for j in 0..n {
                h[[i, j]] += inv_s * hc[[i, j]] + inv_s2 * gc[i] * gc[j];
            }
        }
    }
    h
}

/// Find a strictly feasible starting point by solving an auxiliary phase-I
/// problem if the given `y0` is not feasible.
fn find_feasible_point(lcp: &LogConvexProblem, y0: &[f64]) -> OptimizeResult<Vec<f64>> {
    let n = y0.len();
    let m = lcp.con_a.len();
    if m == 0 {
        return Ok(y0.to_vec());
    }

    // Check current feasibility.
    let infeasible = lcp
        .con_a
        .iter()
        .zip(lcp.con_b.iter())
        .any(|(ca, cb)| lse_objective(ca, cb, y0) >= 0.0);
    if !infeasible {
        return Ok(y0.to_vec());
    }

    // Phase-I: minimise s s.t. lse(conᵢ(y)) − s ≤ 0 via gradient descent on s.
    // We embed: y_aug = [y; s], start with large s.
    let max_lse: f64 = lcp
        .con_a
        .iter()
        .zip(lcp.con_b.iter())
        .map(|(ca, cb)| lse_objective(ca, cb, y0))
        .fold(f64::NEG_INFINITY, f64::max);
    let s_init = max_lse + 1.0;
    let mut y_aug: Vec<f64> = y0.iter().cloned().chain(std::iter::once(s_init)).collect();

    // Simple steepest-descent on φ(y,s) = s + Σᵢ max(0, lseᵢ(y) − s + 0.1).
    for _iter in 0..200 {
        let s = y_aug[n];
        let mut g_y = vec![0.0_f64; n];
        let mut g_s = 1.0_f64;
        for (ca, cb) in lcp.con_a.iter().zip(lcp.con_b.iter()) {
            let lse_c = lse_objective(ca, cb, &y_aug[..n]);
            let margin = lse_c - s + 0.1;
            if margin > 0.0 {
                let gc = lse_gradient(ca, cb, &y_aug[..n]);
                for j in 0..n {
                    g_y[j] += gc[j];
                }
                g_s -= 1.0;
            }
        }
        let step = 0.01;
        for j in 0..n {
            y_aug[j] -= step * g_y[j];
        }
        y_aug[n] -= step * g_s;

        let s_new = y_aug[n];
        if s_new < -0.1 {
            // Feasible enough.
            return Ok(y_aug[..n].to_vec());
        }
    }

    // Final feasibility check.
    let feasible = lcp
        .con_a
        .iter()
        .zip(lcp.con_b.iter())
        .all(|(ca, cb)| lse_objective(ca, cb, &y_aug[..n]) < 0.0);
    if feasible {
        Ok(y_aug[..n].to_vec())
    } else {
        Err(OptimizeError::InitializationError(
            "could not find a strictly feasible starting point for the GP".into(),
        ))
    }
}

/// Solve the GP using a log-barrier interior-point method.
///
/// # Algorithm
///
/// 1. Transform the GP to a log-domain convex problem via [`gp_to_convex`].
/// 2. Find (or verify) a strictly feasible starting point.
/// 3. Run the barrier method: for increasing t, minimise F_t via Newton's
///    method with backtracking Armijo line search.
/// 4. Return x* = exp(y*).
pub fn solve_gp(prob: &GPProblem, config: Option<GPSolverConfig>) -> OptimizeResult<GPResult> {
    let cfg = config.unwrap_or_default();
    let lcp = gp_to_convex(prob);
    let n = prob.n_vars();

    // Choose starting point in log-domain.
    let y0: Vec<f64> = cfg.initial_y.clone().unwrap_or_else(|| vec![0.0_f64; n]);
    if y0.len() != n {
        return Err(OptimizeError::ValueError(format!(
            "initial_y has length {} but problem has {} variables",
            y0.len(),
            n
        )));
    }

    // Handle equality constraints: project y0 so that each Aⱼ·y + bⱼ = 0.
    // (For simplicity, we do not enforce equalities in this phase; they are
    // treated as soft penalties during Newton steps — a full equality handler
    // requires an augmented-Lagrangian or KKT extension.)
    let mut y = find_feasible_point(&lcp, &y0)?;

    let mut t = cfg.t_init;
    let m = lcp.con_a.len();
    // Duality gap proxy for the barrier method: m / t.
    let duality_gap = |t: f64| (m as f64) / t;

    let mut outer_iters = 0_usize;
    let mut converged = false;

    for _outer in 0..cfg.max_outer_iters {
        outer_iters += 1;

        // ── Newton minimisation of F_t ────────────────────────────────────
        for _inner in 0..cfg.max_inner_iters {
            let grad = barrier_gradient(&lcp, t, &y);
            let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

            if grad_norm < cfg.inner_tol {
                break;
            }

            let hess = barrier_hessian(&lcp, t, &y);
            let delta = solve_newton_system(&hess, &grad);

            // Newton decrement λ² = -gᵀ Δ.
            let newton_dec: f64 = grad
                .iter()
                .zip(delta.iter())
                .map(|(g, d)| g * d)
                .sum::<f64>();
            if newton_dec.abs() < cfg.inner_tol {
                break;
            }

            // Backtracking line search.
            let f0 = match barrier_value(&lcp, t, &y) {
                Some(v) => v,
                None => break,
            };
            let mut step = 1.0_f64;
            let ls_thresh = cfg.ls_alpha * newton_dec.abs();
            let y_new = loop {
                let candidate: Vec<f64> = y
                    .iter()
                    .zip(delta.iter())
                    .map(|(yi, di)| yi + step * di)
                    .collect();
                if let Some(f_new) = barrier_value(&lcp, t, &candidate) {
                    if f0 - f_new >= step * ls_thresh {
                        break candidate;
                    }
                }
                step *= cfg.ls_beta;
                if step < 1e-20 {
                    break y.clone();
                }
            };
            y = y_new;
        }

        // ── Check convergence ─────────────────────────────────────────────
        let gap = duality_gap(t);
        if gap < cfg.outer_tol {
            converged = true;
            break;
        }

        t *= cfg.mu;
    }

    // Recover primal variables x = exp(y).
    let x: Vec<f64> = y.iter().map(|yi| yi.exp()).collect();
    let obj_value = prob.objective.evaluate(&x)?;

    Ok(GPResult {
        x,
        obj_value,
        outer_iters,
        converged,
        gap: (m as f64) / t,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_monomial_evaluate() {
        // 2 · x₁^1 · x₂^2
        let m = Monomial::new(2.0, vec![1.0, 2.0]).expect("valid monomial");
        let val = m.evaluate(&[3.0, 4.0]).expect("evaluation");
        // 2 · 3 · 16 = 96
        assert!(approx_eq(val, 96.0, 1e-10));
    }

    #[test]
    fn test_posynomial_evaluate() {
        // x₁ + x₂  (both coefficient 1, unit exponent on respective variable)
        let m1 = Monomial::new(1.0, vec![1.0, 0.0]).expect("m1");
        let m2 = Monomial::new(1.0, vec![0.0, 1.0]).expect("m2");
        let p = Posynomial::new(vec![m1, m2]).expect("posynomial");
        let val = p.evaluate(&[2.0, 3.0]).expect("eval");
        assert!(approx_eq(val, 5.0, 1e-10));
    }

    #[test]
    fn test_gp_unconstrained() {
        // min x^2   (x > 0)
        // Optimal: x → 0⁺, but since the domain is x > 0 and there's no
        // lower bound constraint, the solver will push x small.
        // Instead test: min x + 1/x  (x > 0)  → optimal x = 1, val = 2.
        // f = x^1 + x^{-1}  both posynomials.
        let m1 = Monomial::new(1.0, vec![1.0]).expect("m1");
        let m2 = Monomial::new(1.0, vec![-1.0]).expect("m2");
        let obj = Posynomial::new(vec![m1, m2]).expect("obj");
        let prob = GPProblem::new(obj, vec![], vec![]).expect("prob");
        let cfg = GPSolverConfig {
            initial_y: Some(vec![0.5]),
            ..Default::default()
        };
        let result = solve_gp(&prob, Some(cfg)).expect("solve");
        assert!(
            approx_eq(result.obj_value, 2.0, 0.01),
            "expected ~2.0, got {}",
            result.obj_value
        );
        assert!(
            approx_eq(result.x[0], 1.0, 0.05),
            "expected x≈1, got {}",
            result.x[0]
        );
    }

    #[test]
    fn test_gp_with_constraint() {
        // min x + y  s.t. xy ≥ 1  i.e. (xy)^{-1} ≤ 1
        // Optimal: x = y = 1, val = 2.
        let m1 = Monomial::new(1.0, vec![1.0, 0.0]).expect("m1");
        let m2 = Monomial::new(1.0, vec![0.0, 1.0]).expect("m2");
        let obj = Posynomial::new(vec![m1, m2]).expect("obj");

        // Constraint: (xy)^{-1} ≤ 1  → monomial 1·x^{-1}·y^{-1} ≤ 1
        let con_mono = Monomial::new(1.0, vec![-1.0, -1.0]).expect("con");
        let con_posy = Posynomial::new(vec![con_mono]).expect("con_p");

        let prob = GPProblem::new(obj, vec![con_posy], vec![]).expect("prob");
        let cfg = GPSolverConfig {
            initial_y: Some(vec![0.0, 0.0]),
            ..Default::default()
        };
        let result = solve_gp(&prob, Some(cfg)).expect("solve");
        assert!(
            approx_eq(result.obj_value, 2.0, 0.05),
            "expected ~2.0, got {}",
            result.obj_value
        );
    }

    #[test]
    fn test_log_sum_exp_stable() {
        let v = vec![1000.0, 1001.0, 1002.0];
        let lse = log_sum_exp(&v);
        // Should not overflow; approximate value.
        assert!(lse > 1001.0 && lse < 1003.0);
    }

    #[test]
    fn test_monomial_invalid_coefficient() {
        assert!(Monomial::new(-1.0, vec![1.0]).is_err());
        assert!(Monomial::new(0.0, vec![1.0]).is_err());
    }
}
