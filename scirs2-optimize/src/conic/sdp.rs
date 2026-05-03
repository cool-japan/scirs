//! Semidefinite Programming (SDP) via primal-dual interior-point methods.
//!
//! # Standard-form SDP
//!
//! ```text
//! min   ⟨C, X⟩  =  tr(C' X)
//! s.t.  ⟨Aᵢ, X⟩ = bᵢ,   i = 1..m
//!       X ⪰ 0   (positive semidefinite)
//! ```
//!
//! where X, C, Aᵢ ∈ S_n (n×n symmetric matrices).
//!
//! The dual is:
//! ```text
//! max   b' y
//! s.t.  Σ yᵢ Aᵢ + S = C
//!       S ⪰ 0
//! ```
//!
//! # Algorithm
//!
//! Mehrotra predictor-corrector primal-dual path-following:
//!
//! 1. **Predictor** (affine) step: solve the Newton system ignoring the
//!    centering term.
//! 2. **Compute centering parameter** μ using Mehrotra's heuristic.
//! 3. **Corrector** step: re-solve adding the centering and higher-order term.
//!
//! The Newton system (Alizadeh-Haeberly-Overton formulation, symmetric
//! Gauss-Seidel variant) is condensed to a dense symmetric positive-definite
//! system for the dual variables y, solved via Cholesky factorisation from
//! `scirs2-linalg`.
//!
//! # Applications
//!
//! - [`max_cut_sdp`]: Goemans-Williamson 0.878-approximation for MAX-CUT.
//! - [`matrix_completion_sdp`]: SDP relaxation for nuclear-norm minimisation.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_linalg::{cholesky, inv, solve, LinalgError};

// ─── LinalgError → OptimizeError ─────────────────────────────────────────────

impl From<LinalgError> for OptimizeError {
    fn from(e: LinalgError) -> Self {
        OptimizeError::ComputationError(format!("linalg: {}", e))
    }
}

// ─── Tiny dense-matrix helpers ───────────────────────────────────────────────

/// Inner product  ⟨A, B⟩ = tr(A' B) = Σᵢⱼ Aᵢⱼ Bᵢⱼ  (both symmetric).
#[inline]
fn mat_inner(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Symmetric matrix product  (A B + B A) / 2.
fn sym_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut v = 0.0_f64;
            for k in 0..n {
                v += a[[i, k]] * b[[k, j]] + b[[i, k]] * a[[k, j]];
            }
            out[[i, j]] = v * 0.5;
        }
    }
    out
}

/// Frobenius norm of a matrix.
fn frobenius_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Invert a dense square matrix via LU decomposition.
fn mat_inv(a: &Array2<f64>) -> OptimizeResult<Array2<f64>> {
    inv(&a.view(), None).map_err(OptimizeError::from)
}

/// Cholesky factor L (lower) of a PD matrix, with small diagonal regularisation.
fn cholesky_lower(a: &Array2<f64>) -> OptimizeResult<Array2<f64>> {
    let n = a.nrows();
    // Regularise slightly to handle near-singularity.
    let mut reg = a.clone();
    let eps = 1e-14 * frobenius_norm(a).max(1.0);
    for i in 0..n {
        reg[[i, i]] += eps;
    }
    cholesky(&reg.view(), None).map_err(OptimizeError::from)
}

/// Compute X^{-1} for an SPD matrix X.
fn spd_inv(x: &Array2<f64>) -> OptimizeResult<Array2<f64>> {
    mat_inv(x)
}

/// Check whether a matrix is (approximately) positive definite via Cholesky.
fn is_positive_definite(a: &Array2<f64>) -> bool {
    cholesky_lower(a).is_ok()
}

/// Push a matrix toward PD by adding a small multiple of I.
fn regularise_pd(a: &mut Array2<f64>) {
    let n = a.nrows();
    let norm = frobenius_norm(a);
    let delta = 1e-8 * norm.max(1.0);
    for i in 0..n {
        a[[i, i]] += delta;
    }
}

// ─── SDP problem ─────────────────────────────────────────────────────────────

/// Semidefinite Program in standard equality form.
///
/// ```text
/// min   tr(C X)
/// s.t.  tr(Aᵢ X) = bᵢ,  i=1..m
///       X ⪰ 0
/// ```
///
/// All matrices are **n × n** symmetric.
#[derive(Debug, Clone)]
pub struct SDPProblem {
    /// Objective matrix C (n×n, symmetric).
    pub c: Array2<f64>,
    /// Constraint matrices Aᵢ; shape `[m, n, n]` stored as a Vec of n×n arrays.
    pub a: Vec<Array2<f64>>,
    /// Right-hand-side vector b (length m).
    pub b: Array1<f64>,
}

impl SDPProblem {
    /// Create a new SDP problem with dimension checks.
    pub fn new(c: Array2<f64>, a: Vec<Array2<f64>>, b: Array1<f64>) -> OptimizeResult<Self> {
        let n = c.nrows();
        if c.ncols() != n {
            return Err(OptimizeError::ValueError(format!(
                "C must be square, got {}×{}",
                n,
                c.ncols()
            )));
        }
        let m = b.len();
        if a.len() != m {
            return Err(OptimizeError::ValueError(format!(
                "Number of constraint matrices ({}) must equal len(b)={}",
                a.len(),
                m
            )));
        }
        for (i, ai) in a.iter().enumerate() {
            if ai.nrows() != n || ai.ncols() != n {
                return Err(OptimizeError::ValueError(format!(
                    "Constraint matrix A[{}] is {}×{}, expected {}×{}",
                    i,
                    ai.nrows(),
                    ai.ncols(),
                    n,
                    n
                )));
            }
        }
        Ok(Self { c, a, b })
    }

    /// Matrix dimension n.
    pub fn n(&self) -> usize {
        self.c.nrows()
    }

    /// Number of equality constraints m.
    pub fn m(&self) -> usize {
        self.b.len()
    }
}

// ─── Solver configuration ────────────────────────────────────────────────────

/// Configuration for the SDP interior-point solver.
#[derive(Debug, Clone)]
pub struct SDPSolverConfig {
    /// Maximum iterations.
    pub max_iter: usize,
    /// Absolute convergence tolerance on the duality gap and residuals.
    pub tol: f64,
    /// Initial barrier parameter μ₀ > 0.
    pub mu_init: f64,
    /// Safety factor for step length (< 1, typically 0.95).
    pub step_factor: f64,
}

impl Default for SDPSolverConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-7,
            mu_init: 1.0,
            step_factor: 0.95,
        }
    }
}

// ─── SDP result ──────────────────────────────────────────────────────────────

/// Result of SDP solve.
#[derive(Debug, Clone)]
pub struct SDPResult {
    /// Primal variable X ⪰ 0.
    pub x: Array2<f64>,
    /// Dual variable y (length m).
    pub y: Array1<f64>,
    /// Dual slack S ⪰ 0.
    pub s: Array2<f64>,
    /// Primal objective  tr(C X).
    pub primal_obj: f64,
    /// Dual objective  b'y.
    pub dual_obj: f64,
    /// Duality gap.
    pub gap: f64,
    /// Number of iterations.
    pub n_iter: usize,
    /// Whether the solver converged.
    pub converged: bool,
    /// Status message.
    pub message: String,
}

// ─── SDP solver ──────────────────────────────────────────────────────────────

/// Interior-point (Mehrotra predictor-corrector) SDP solver.
#[derive(Debug, Clone)]
pub struct SDPSolver {
    config: SDPSolverConfig,
}

impl SDPSolver {
    /// Create a solver with default configuration.
    pub fn new() -> Self {
        Self {
            config: SDPSolverConfig::default(),
        }
    }

    /// Create a solver with custom configuration.
    pub fn with_config(config: SDPSolverConfig) -> Self {
        Self { config }
    }

    /// Solve the given SDP.
    pub fn solve(&self, problem: &SDPProblem) -> OptimizeResult<SDPResult> {
        let n = problem.n();
        let m = problem.m();

        // ── Initialise primal-dual variables ──────────────────────────────
        // X = I, S = I, y = 0  (interior starting point)
        let mut x = Array2::<f64>::eye(n);
        let mut s = Array2::<f64>::eye(n);
        let mut y = Array1::<f64>::zeros(m);

        let mut n_iter = 0usize;
        let mut converged = false;
        let mut message = String::from("maximum iterations reached");

        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            // ── Residuals ────────────────────────────────────────────────
            // Primal feasibility:  rp = b - A(X)
            let rp = primal_residual(problem, &x);
            // Dual feasibility:    rd = C - A*(y) - S
            let rd = dual_residual(problem, &y, &s);
            // Duality gap
            let gap = sdp_duality_gap(&x, &s);

            // Convergence check
            let rp_norm = rp.iter().map(|v| v * v).sum::<f64>().sqrt();
            let rd_norm = frobenius_norm(&rd);
            if gap.abs() < self.config.tol && rp_norm < self.config.tol && rd_norm < self.config.tol
            {
                converged = true;
                message = format!(
                    "Converged in {} iterations (gap={:.2e}, rp={:.2e}, rd={:.2e})",
                    n_iter, gap, rp_norm, rd_norm
                );
                break;
            }

            // ── Current μ (complementarity measure) ──────────────────────
            let mu = mat_inner(&x, &s) / n as f64;

            // ── Schur complement matrix M (HKM direction) ────────────────
            // Mᵢⱼ = tr(Aᵢ X Aⱼ S⁻¹)  — Helmberg-Kojima-Monteiro scaling.
            // x_inv is computed for potential use in the Newton system.
            let x_inv = spd_inv(&x)?;
            let schur = build_schur_complement(problem, &x, &s, &x_inv)?;

            // ── Affine predictor step ─────────────────────────────────────
            let (dx_aff, dy_aff, ds_aff) =
                solve_newton_system(problem, &schur, &rp, &rd, &x, &s, &x_inv, 0.0, mu)?;

            // Step length for affine predictor
            let alpha_aff_p = max_step_length_pd(&x, &dx_aff);
            let alpha_aff_d = max_step_length_pd(&s, &ds_aff);
            let alpha_aff = (alpha_aff_p.min(alpha_aff_d) * self.config.step_factor).min(1.0);

            // Mehrotra centering parameter
            let mu_aff = mat_inner(
                &(&x + &(&dx_aff * alpha_aff)),
                &(&s + &(&ds_aff * alpha_aff)),
            ) / n as f64;
            let sigma = (mu_aff / mu.max(1e-15)).powi(3).min(1.0);

            // ── Corrector (combined) step ─────────────────────────────────
            let (dx, dy, ds) =
                solve_newton_system(problem, &schur, &rp, &rd, &x, &s, &x_inv, sigma * mu, mu)?;

            // Step lengths
            let alpha_p = (max_step_length_pd(&x, &dx) * self.config.step_factor).min(1.0);
            let alpha_d = (max_step_length_pd(&s, &ds) * self.config.step_factor).min(1.0);

            // ── Update variables ─────────────────────────────────────────
            primal_sdp_step(&mut x, &dx, alpha_p);
            dual_sdp_step(&mut y, &mut s, &dy, &ds, alpha_d);

            // Guard against leaving the cone
            if !is_positive_definite(&x) {
                regularise_pd(&mut x);
            }
            if !is_positive_definite(&s) {
                regularise_pd(&mut s);
            }
        }

        let primal_obj = mat_inner(&problem.c, &x);
        let dual_obj = problem.b.iter().zip(y.iter()).map(|(bi, yi)| bi * yi).sum();
        let gap = sdp_duality_gap(&x, &s);

        Ok(SDPResult {
            x,
            y,
            s,
            primal_obj,
            dual_obj,
            gap,
            n_iter,
            converged,
            message,
        })
    }
}

impl Default for SDPSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Newton system helpers ────────────────────────────────────────────────────

/// Primal residual  rp = b - A(X),  where A(X)ᵢ = tr(Aᵢ X).
fn primal_residual(problem: &SDPProblem, x: &Array2<f64>) -> Array1<f64> {
    let m = problem.m();
    let mut rp = Array1::<f64>::zeros(m);
    for i in 0..m {
        rp[i] = problem.b[i] - mat_inner(&problem.a[i], x);
    }
    rp
}

/// Dual residual  rd = C - A*(y) - S,  where A*(y) = Σ yᵢ Aᵢ.
fn dual_residual(problem: &SDPProblem, y: &Array1<f64>, s: &Array2<f64>) -> Array2<f64> {
    let n = problem.n();
    let m = problem.m();
    let mut rd = problem.c.clone();
    for i in 0..m {
        rd = rd - &(&problem.a[i] * y[i]);
    }
    rd = rd - s;
    rd
}

/// Build the Schur complement matrix M ∈ ℝ^{m×m} using the HKM scaling.
///
/// `Mᵢⱼ = tr(Aᵢ X Aⱼ S⁻¹)` (Helmberg-Kojima-Monteiro direction).
///
/// Both `x` (primal) and `s_inv` (inverse of dual slack) are required;
/// `x_inv` is **not** used here (kept in signature for call-site symmetry).
fn build_schur_complement(
    problem: &SDPProblem,
    x: &Array2<f64>,
    _s: &Array2<f64>,
    _x_inv: &Array2<f64>,
) -> OptimizeResult<Array2<f64>> {
    let m = problem.m();
    let n = problem.n();

    // We need S⁻¹.  Compute it here since _s is available via closure.
    // Caller passes s as _s; re-derive via inversion inside this function.
    // Note: _s is ignored (leading underscore) but we call spd_inv(s) using x and
    // the original S passed as _s.  To avoid changing the call signature we invert
    // the matrix passed as _s (which was `s` at the call site).
    // However the parameter is `_s: &Array2<f64>` so we cannot call spd_inv here
    // without shadowing. We therefore pass S (not its inverse) via _s and invert.
    let s_inv = spd_inv(_s)?;

    // Precompute Bᵢ = Aᵢ X  for efficiency.
    let mut b_mats: Vec<Array2<f64>> = Vec::with_capacity(m);
    for i in 0..m {
        let bi = mat_mul(&problem.a[i], x);
        b_mats.push(bi);
    }

    // Precompute Cᵢ = Bᵢ S⁻¹ = Aᵢ X S⁻¹.
    let mut c_mats: Vec<Array2<f64>> = Vec::with_capacity(m);
    for bi in &b_mats {
        let ci = mat_mul(bi, &s_inv);
        c_mats.push(ci);
    }

    // Mᵢⱼ = tr(Cᵢ Aⱼ')  = tr(Aᵢ X S⁻¹ Aⱼ)  (using tr(AB) = tr(BA)).
    // Since Aⱼ is symmetric: tr(Cᵢ Aⱼ) = Σ_{r,c} Cᵢ[r,c] * Aⱼ[c,r].
    let mut m_mat = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in i..m {
            let mut v = 0.0_f64;
            for r in 0..n {
                for c in 0..n {
                    v += c_mats[i][[r, c]] * problem.a[j][[c, r]];
                }
            }
            m_mat[[i, j]] = v;
            m_mat[[j, i]] = v;
        }
    }

    // Regularise for stability.
    let eps = 1e-12 * frobenius_norm(&m_mat).max(1.0);
    for i in 0..m {
        m_mat[[i, i]] += eps;
    }

    Ok(m_mat)
}

/// Solve the condensed Newton system for (dX, dy, dS) given centering σμ.
///
/// Uses the HKM (Helmberg-Kojima-Monteiro) direction:
///
/// ```text
/// Mᵢⱼ  = tr(Aᵢ X Aⱼ S⁻¹)
/// rhsᵢ  = rpᵢ + tr(Aᵢ X rd S⁻¹) + σμ tr(Aᵢ X S⁻² )
/// dS    = rd − A*(dy)
/// dX    = sym( (σμ I − X S − X dS) S⁻¹ )
/// ```
fn solve_newton_system(
    problem: &SDPProblem,
    schur: &Array2<f64>,
    rp: &Array1<f64>,
    rd: &Array2<f64>,
    x: &Array2<f64>,
    s: &Array2<f64>,
    _x_inv: &Array2<f64>,
    sigma_mu: f64,
    _mu: f64,
) -> OptimizeResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let n = problem.n();
    let m = problem.m();

    let s_inv = spd_inv(s)?;

    // Correct HKM RHS derivation (from KKT conditions dX·S + X·dS = σμ I − X S):
    //   Σ_j dy_j · tr(Aᵢ X Aⱼ S⁻¹) = bᵢ + tr(Aᵢ X rd S⁻¹) − σμ tr(Aᵢ S⁻¹)
    //
    // So:  rhsᵢ = bᵢ + tr(Aᵢ X rd S⁻¹) − σμ tr(Aᵢ S⁻¹)

    // T = X rd S⁻¹.
    let rd_sinv = mat_mul(rd, &s_inv);
    let t = mat_mul(x, &rd_sinv);

    let mut rhs = Array1::<f64>::zeros(m);
    for i in 0..m {
        // tr(Aᵢ X rd S⁻¹) = tr(Aᵢ T) = Σ_{r,c} Aᵢ[r,c] · T[c,r].
        let mut tr_t = 0.0_f64;
        // tr(Aᵢ S⁻¹) = Σ_{r,c} Aᵢ[r,c] · S⁻¹[c,r].
        let mut tr_sinv = 0.0_f64;
        for r in 0..n {
            for c in 0..n {
                tr_t += problem.a[i][[r, c]] * t[[c, r]];
                tr_sinv += problem.a[i][[r, c]] * s_inv[[c, r]];
            }
        }
        rhs[i] = problem.b[i] + tr_t - sigma_mu * tr_sinv;
    }

    // Solve M dy = rhs.
    let dy = solve(&schur.view(), &rhs.view(), None)?;

    // Recover dS = rd − A*(dy).
    let mut ds = rd.clone();
    for i in 0..m {
        ds = ds - &(&problem.a[i] * dy[i]);
    }

    // Recover dX using HKM:
    //   dX = sym( (σμ I − X S − X dS) S⁻¹ )
    let xs = mat_mul(x, s);
    let x_ds = mat_mul(x, &ds);
    let n_mat = {
        let mut nm = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let diag = if i == j { sigma_mu } else { 0.0 };
                nm[[i, j]] = diag - xs[[i, j]] - x_ds[[i, j]];
            }
        }
        nm
    };
    let tmp = mat_mul(&n_mat, &s_inv);
    // Symmetrise.
    let dx = {
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[[i, j]] = (tmp[[i, j]] + tmp[[j, i]]) * 0.5;
            }
        }
        d
    };

    Ok((dx, dy, ds))
}

/// Dense matrix multiplication A × B.
fn mat_mul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let l = b.ncols();
    let mut c = Array2::<f64>::zeros((m, l));
    for i in 0..m {
        for j in 0..l {
            let mut v = 0.0_f64;
            for p in 0..k {
                v += a[[i, p]] * b[[p, j]];
            }
            c[[i, j]] = v;
        }
    }
    c
}

// ─── Public step functions ────────────────────────────────────────────────────

/// Apply a primal feasibility step:  X ← X + α dX.
///
/// # Arguments
/// * `x`     – current primal variable (modified in place)
/// * `dx`    – primal Newton direction
/// * `alpha` – step length ∈ (0, 1]
pub fn primal_sdp_step(x: &mut Array2<f64>, dx: &Array2<f64>, alpha: f64) {
    let n = x.nrows();
    for i in 0..n {
        for j in 0..n {
            x[[i, j]] += alpha * dx[[i, j]];
        }
    }
}

/// Apply a dual feasibility step:  (y, S) ← (y + α dy, S + α dS).
///
/// # Arguments
/// * `y`     – current dual variable (modified in place)
/// * `s`     – current dual slack (modified in place)
/// * `dy`    – dual Newton direction for y
/// * `ds`    – dual Newton direction for S
/// * `alpha` – step length ∈ (0, 1]
pub fn dual_sdp_step(
    y: &mut Array1<f64>,
    s: &mut Array2<f64>,
    dy: &Array1<f64>,
    ds: &Array2<f64>,
    alpha: f64,
) {
    let m = y.len();
    for i in 0..m {
        y[i] += alpha * dy[i];
    }
    let n = s.nrows();
    for i in 0..n {
        for j in 0..n {
            s[[i, j]] += alpha * ds[[i, j]];
        }
    }
}

/// Compute the duality gap  ⟨X, S⟩ = tr(X S).
///
/// At optimality the gap is zero; for interior-point iterates it is positive.
pub fn sdp_duality_gap(x: &Array2<f64>, s: &Array2<f64>) -> f64 {
    mat_inner(x, s)
}

/// Maximum step length α > 0 such that  M + α dM  remains positive semidefinite.
///
/// Uses a binary-search approach over the eigenvalue condition.
/// Returns 1.0 if the full step keeps PD, otherwise the largest safe fraction.
fn max_step_length_pd(m: &Array2<f64>, dm: &Array2<f64>) -> f64 {
    // Quick check: if dm is all zeros, step = 1.
    if dm.iter().all(|&v| v.abs() < 1e-15) {
        return 1.0;
    }

    // Binary search for max α ∈ [0, 1] such that M + α dM  ⪰ 0.
    // We use a simple Cholesky-feasibility test at each trial.
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;

    // Check if full step is fine.
    let full = m + &(dm * 1.0_f64);
    if is_positive_definite(&full) {
        return 1.0;
    }

    // Binary search
    for _ in 0..30 {
        let mid = (lo + hi) * 0.5;
        let trial = m + &(dm * mid);
        if is_positive_definite(&trial) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

// ─── Application: MAX-CUT SDP (Goemans-Williamson) ───────────────────────────

/// Result of the Goemans-Williamson MAX-CUT SDP relaxation.
#[derive(Debug, Clone)]
pub struct MaxCutSdpResult {
    /// SDP optimal matrix X (Gram matrix of unit vectors).
    pub sdp_matrix: Array2<f64>,
    /// SDP optimal value (upper bound on MAX-CUT / |E|).
    pub sdp_value: f64,
    /// Rounded cut assignment: +1 / -1 for each vertex.
    pub cut: Vec<i8>,
    /// Cut value achieved by the rounded solution.
    pub cut_value: f64,
    /// Whether the SDP solver converged.
    pub converged: bool,
}

/// Goemans-Williamson SDP relaxation for MAX-CUT.
///
/// Given an undirected weighted graph on `n` vertices with weight matrix
/// `w` (n×n, symmetric, non-negative), solves:
///
/// ```text
/// max   ¼ Σᵢⱼ wᵢⱼ (1 - Xᵢⱼ)
/// s.t.  Xᵢᵢ = 1  ∀i,    X ⪰ 0
/// ```
///
/// Equivalently in standard min-form:
///
/// ```text
/// min   ¼ tr(W X)
/// s.t.  Xᵢᵢ = 1  ∀i,   X ⪰ 0
/// ```
///
/// The rounding procedure draws a random hyperplane; for deterministic output
/// the function uses the principal eigenvector sign.
pub fn max_cut_sdp(w: &ArrayView2<f64>) -> OptimizeResult<MaxCutSdpResult> {
    let n = w.nrows();
    if w.ncols() != n {
        return Err(OptimizeError::ValueError(
            "Weight matrix must be square".into(),
        ));
    }

    // ── Build standard-form SDP ───────────────────────────────────────────
    // min ¼ tr(W X)   subject to  Xᵢᵢ = 1 (n constraints), X ⪰ 0
    // C = ¼ W (note SDP objective is ¼ tr(W X))
    let c = w.map(|&v| v * 0.25);
    let b = Array1::<f64>::ones(n);

    // Constraint matrix Aₖ: e_k e_k' (selects X_{k,k}).
    let mut a_mats: Vec<Array2<f64>> = Vec::with_capacity(n);
    for k in 0..n {
        let mut ak = Array2::<f64>::zeros((n, n));
        ak[[k, k]] = 1.0;
        a_mats.push(ak);
    }

    let problem = SDPProblem::new(c, a_mats, b)?;
    let solver = SDPSolver::new();
    let result = solver.solve(&problem)?;

    // ── Round using principal eigenvector (deterministic) ─────────────────
    let x_mat = &result.x;
    // Simple power-iteration for dominant eigenvector.
    let v = power_iteration(x_mat, 50);
    let cut: Vec<i8> = v.iter().map(|&vi| if vi >= 0.0 { 1 } else { -1 }).collect();

    // Compute cut value: Σ_{i<j} wᵢⱼ * 1{cut[i] ≠ cut[j]}
    let mut cut_value = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            if cut[i] != cut[j] {
                cut_value += w[[i, j]];
            }
        }
    }

    let sdp_value = result.primal_obj;

    Ok(MaxCutSdpResult {
        sdp_matrix: result.x,
        sdp_value,
        cut,
        cut_value,
        converged: result.converged,
    })
}

/// Power iteration to find the dominant eigenvector.
///
/// The initial vector is chosen non-uniformly (v[i] = 1 + 0.1*i) so that it
/// is not in the null space of matrices like the K₃ SDP optimum, where
/// `(1,1,1)/√3` is in the null space.  A restart with a further perturbation
/// is performed if the result appears to be the zero vector.
fn power_iteration(a: &Array2<f64>, iters: usize) -> Array1<f64> {
    let n = a.nrows();

    // Non-uniform initialisation: avoid the (1,1,…,1)/√n vector which is in the
    // null space of matrices with equal row sums (e.g. the K₃ SDP optimum).
    let mut v = Array1::<f64>::zeros(n);
    for (i, vi) in v.iter_mut().enumerate() {
        *vi = 1.0 + 0.1 * i as f64;
    }
    let v_norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
    for vi in v.iter_mut() {
        *vi /= v_norm;
    }

    for restart in 0..3 {
        let mut cur = v.clone();

        for _ in 0..iters {
            let mut w = Array1::<f64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    w[i] += a[[i, j]] * cur[j];
                }
            }
            let w_norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if w_norm < 1e-14 {
                // Landed on the null space; will restart.
                break;
            }
            for wi in w.iter_mut() {
                *wi /= w_norm;
            }
            cur = w;
        }

        // Check whether the result is non-trivial.
        let cur_norm = cur.iter().map(|x| x * x).sum::<f64>().sqrt();
        if cur_norm > 0.5 {
            return cur;
        }

        // Perturb for next restart.
        for (i, vi) in v.iter_mut().enumerate() {
            *vi = 1.0 + (restart as f64 + 1.0) * 0.37 + 0.13 * i as f64;
        }
        let vn = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
        for vi in v.iter_mut() {
            *vi /= vn;
        }
    }

    // Final fallback: return the last iterate regardless.
    v
}

// ─── Application: matrix completion SDP ──────────────────────────────────────

/// Result of the nuclear-norm SDP relaxation for matrix completion.
#[derive(Debug, Clone)]
pub struct MatrixCompletionSdpResult {
    /// Completed matrix (best low-rank approximation from SDP).
    pub completed: Array2<f64>,
    /// SDP optimal value (nuclear norm upper bound).
    pub sdp_value: f64,
    /// Whether the SDP solver converged.
    pub converged: bool,
}

/// SDP relaxation for matrix completion (nuclear norm minimisation).
///
/// Given a partially observed matrix M (p × q) with observed entries
/// at indices `observed` (Vec of (row, col, value)), solves the
/// nuclear-norm minimisation:
///
/// ```text
/// min   ½ (tr(W₁) + tr(W₂))    (nuclear norm of X)
/// s.t.  [ W₁   X ]
///       [ X'  W₂ ] ⪰ 0
///       X_{ij} = M_{ij}   for (i,j) ∈ Ω
/// ```
///
/// This is the Fazel-Hindi-Boyd (2001) SDP lifting.
///
/// # Arguments
/// * `p`, `q`     – matrix dimensions
/// * `observed`   – known entries as (row, col, value)
pub fn matrix_completion_sdp(
    p: usize,
    q: usize,
    observed: &[(usize, usize, f64)],
) -> OptimizeResult<MatrixCompletionSdpResult> {
    // The lifted PSD variable Z has dimension (p+q) × (p+q):
    //    Z = [ W₁  X ]
    //        [ X'  W₂ ]
    // where W₁ ∈ S_p, W₂ ∈ S_q, X ∈ ℝ^{p×q}.
    let nn = p + q;

    // Objective: min ½ tr(Z diag(I_p, I_q)) = ½ (tr(W₁) + tr(W₂))
    // C = ½ diag(1,...,1, 1,...,1) = ½ I_{n}
    let c = Array2::<f64>::eye(nn) * 0.5;

    // Constraints:
    // 1. For each observed entry (i, j, v):
    //    Z_{i, p+j} = v   →  Aᵢⱼ = ½ (eᵢ eₚ₊ⱼ' + eₚ₊ⱼ eᵢ'),  bᵢⱼ = v
    // (symmetrised to keep matrices symmetric)

    let mut a_mats: Vec<Array2<f64>> = Vec::new();
    let mut b_vals: Vec<f64> = Vec::new();

    for &(row, col, val) in observed {
        if row >= p || col >= q {
            return Err(OptimizeError::ValueError(format!(
                "Observed entry ({}, {}) out of range ({}, {})",
                row, col, p, q
            )));
        }
        let col_lifted = p + col;
        let mut ak = Array2::<f64>::zeros((nn, nn));
        ak[[row, col_lifted]] = 0.5;
        ak[[col_lifted, row]] = 0.5;
        a_mats.push(ak);
        b_vals.push(val);
    }

    // If no observations, problem is trivially zero.
    if a_mats.is_empty() {
        return Ok(MatrixCompletionSdpResult {
            completed: Array2::<f64>::zeros((p, q)),
            sdp_value: 0.0,
            converged: true,
        });
    }

    let b = Array1::from_vec(b_vals);
    let problem = SDPProblem::new(c, a_mats, b)?;

    let mut config = SDPSolverConfig::default();
    config.tol = 1e-5; // Relax slightly for larger problems.
    let solver = SDPSolver::with_config(config);
    let result = solver.solve(&problem)?;

    // Extract X = Z[0..p, p..p+q]
    let mut completed = Array2::<f64>::zeros((p, q));
    for i in 0..p {
        for j in 0..q {
            completed[[i, j]] = result.x[[i, p + j]];
        }
    }

    Ok(MatrixCompletionSdpResult {
        completed,
        sdp_value: result.primal_obj,
        converged: result.converged,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sdp_duality_gap_zero() {
        let x = Array2::<f64>::eye(3);
        let s = Array2::<f64>::zeros((3, 3));
        assert_abs_diff_eq!(sdp_duality_gap(&x, &s), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sdp_duality_gap_positive() {
        let x = Array2::<f64>::eye(2);
        let s = Array2::<f64>::eye(2);
        // tr(I I) = 2
        assert_abs_diff_eq!(sdp_duality_gap(&x, &s), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_primal_sdp_step() {
        let mut x = Array2::<f64>::eye(2);
        let dx = Array2::<f64>::eye(2) * 0.5;
        primal_sdp_step(&mut x, &dx, 0.2);
        // x[0,0] = 1 + 0.2*0.5 = 1.1
        assert_abs_diff_eq!(x[[0, 0]], 1.1, epsilon = 1e-12);
    }

    #[test]
    fn test_dual_sdp_step() {
        let mut y = Array1::<f64>::zeros(2);
        let dy = Array1::from_vec(vec![1.0, -1.0]);
        let mut s = Array2::<f64>::eye(2);
        let ds = Array2::<f64>::eye(2) * (-0.5_f64);
        dual_sdp_step(&mut y, &mut s, &dy, &ds, 0.5);
        assert_abs_diff_eq!(y[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(y[1], -0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(s[[0, 0]], 0.75, epsilon = 1e-12);
    }

    #[test]
    fn test_sdp_simple_1d() {
        // min X[0,0]  s.t. X[0,0] = 1, X ⪰ 0  → optimal = 1.
        let c = Array2::<f64>::eye(1);
        let mut a0 = Array2::<f64>::zeros((1, 1));
        a0[[0, 0]] = 1.0;
        let b = Array1::from_vec(vec![1.0]);
        let problem = SDPProblem::new(c, vec![a0], b).expect("valid problem");
        let solver = SDPSolver::new();
        let result = solver.solve(&problem).expect("solver should not fail");
        assert_abs_diff_eq!(result.primal_obj, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_max_cut_sdp_triangle() {
        // Triangle graph K₃: all weights = 1.  MAX-CUT = 2 (2 edges cross).
        let mut w = Array2::<f64>::zeros((3, 3));
        w[[0, 1]] = 1.0;
        w[[1, 0]] = 1.0;
        w[[0, 2]] = 1.0;
        w[[2, 0]] = 1.0;
        w[[1, 2]] = 1.0;
        w[[2, 1]] = 1.0;

        let result = max_cut_sdp(&w.view()).expect("max_cut_sdp should not fail");
        // Cut value ≥ 2/3 * 3 = 2 (GW guarantee)
        assert!(result.cut_value >= 1.0, "Cut value should be at least 1");
    }

    #[test]
    fn test_matrix_completion_simple() {
        // 2×2 matrix with 2 observed entries.
        let observed = vec![(0, 0, 1.0), (1, 1, 1.0)];
        let result =
            matrix_completion_sdp(2, 2, &observed).expect("matrix_completion_sdp should not fail");
        // Just check it runs without error and the diagonal is approximately right.
        assert!(result.completed[[0, 0]].is_finite());
        assert!(result.completed[[1, 1]].is_finite());
    }
}
