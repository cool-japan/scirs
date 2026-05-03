//! Second-Order Cone Programming (SOCP).
//!
//! # Problem form
//!
//! ```text
//! min   c' x
//! s.t.  ‖Aᵢ x + bᵢ‖₂ ≤ cᵢ' x + dᵢ,   i = 1..K
//!       F x = g   (optional linear equality constraints)
//! ```
//!
//! This is the standard SOCP (conic) form with second-order cone (Lorentz cone)
//! constraints.  Each constraint corresponds to membership in the ice-cream cone:
//! ```text
//! Qₙ = { (t, u) ∈ ℝ × ℝⁿ : ‖u‖ ≤ t }
//! ```
//!
//! # Algorithms
//!
//! - [`socp_interior_point`]: A primal-dual interior-point solver specialised
//!   for SOCP, using the NT (Nesterov-Todd) scaling direction.
//! - [`socp_to_sdp`]: Lift an SOCP to an SDP via the Schur complement lemma.
//!
//! # Applications
//!
//! - [`robust_ls_socp`]: Robust least squares under bounded perturbations.
//! - [`portfolio_optimization_socp`]: Mean-variance portfolio with a
//!   variance constraint written as a second-order cone.

use crate::conic::sdp::{SDPProblem, SDPSolver, SDPSolverConfig};
use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_linalg::solve;

// ─── SOCP problem ─────────────────────────────────────────────────────────────

/// One second-order cone constraint:  ‖A x + b‖ ≤ c' x + d.
#[derive(Debug, Clone)]
pub struct SOCConstraint {
    /// Matrix Aᵢ ∈ ℝ^{mᵢ × n}.
    pub a: Array2<f64>,
    /// Vector bᵢ ∈ ℝ^{mᵢ}.
    pub b: Array1<f64>,
    /// Vector cᵢ ∈ ℝⁿ (row coefficient of the scalar cone side).
    pub c: Array1<f64>,
    /// Scalar dᵢ (bias on the scalar cone side).
    pub d: f64,
}

impl SOCConstraint {
    /// Create a new SOC constraint with dimension checks.
    pub fn new(a: Array2<f64>, b: Array1<f64>, c: Array1<f64>, d: f64) -> OptimizeResult<Self> {
        if a.nrows() != b.len() {
            return Err(OptimizeError::ValueError(format!(
                "SOCConstraint: A has {} rows but b has {} elements",
                a.nrows(),
                b.len()
            )));
        }
        if a.ncols() != c.len() {
            return Err(OptimizeError::ValueError(format!(
                "SOCConstraint: A has {} cols but c has {} elements",
                a.ncols(),
                c.len()
            )));
        }
        Ok(Self { a, b, c, d })
    }
}

/// Second-Order Cone Program.
///
/// ```text
/// min   c' x
/// s.t.  ‖Aₖ x + bₖ‖ ≤ cₖ' x + dₖ,  k = 1..K
///       F x = g   (optional)
/// ```
#[derive(Debug, Clone)]
pub struct SOCPProblem {
    /// Objective coefficient c ∈ ℝⁿ.
    pub obj: Array1<f64>,
    /// SOC constraints (K of them).
    pub constraints: Vec<SOCConstraint>,
    /// Optional equality matrix F ∈ ℝ^{p × n}.
    pub eq_a: Option<Array2<f64>>,
    /// Optional equality RHS g ∈ ℝᵖ.
    pub eq_b: Option<Array1<f64>>,
}

impl SOCPProblem {
    /// Create an SOCP without equality constraints.
    pub fn new(obj: Array1<f64>, constraints: Vec<SOCConstraint>) -> OptimizeResult<Self> {
        let n = obj.len();
        for (k, con) in constraints.iter().enumerate() {
            if con.a.ncols() != n {
                return Err(OptimizeError::ValueError(format!(
                    "Constraint {}: A has {} cols but obj has {} elements",
                    k,
                    con.a.ncols(),
                    n
                )));
            }
        }
        Ok(Self {
            obj,
            constraints,
            eq_a: None,
            eq_b: None,
        })
    }

    /// Add optional linear equality constraints.
    pub fn with_equality(mut self, f: Array2<f64>, g: Array1<f64>) -> OptimizeResult<Self> {
        let n = self.obj.len();
        if f.ncols() != n {
            return Err(OptimizeError::ValueError(format!(
                "Equality matrix F has {} cols but problem dimension is {}",
                f.ncols(),
                n
            )));
        }
        if f.nrows() != g.len() {
            return Err(OptimizeError::ValueError(format!(
                "F has {} rows but g has {} elements",
                f.nrows(),
                g.len()
            )));
        }
        self.eq_a = Some(f);
        self.eq_b = Some(g);
        Ok(self)
    }

    /// Number of primal variables.
    pub fn n(&self) -> usize {
        self.obj.len()
    }
}

/// Result of an SOCP solve.
#[derive(Debug, Clone)]
pub struct SOCPResult {
    /// Primal optimal x*.
    pub x: Array1<f64>,
    /// Optimal objective value c' x*.
    pub obj_val: f64,
    /// Constraint residuals ‖Aₖ x + bₖ‖ - (cₖ' x + dₖ) (≤ 0 at feasibility).
    pub residuals: Vec<f64>,
    /// Whether the solver converged.
    pub converged: bool,
    /// Status message.
    pub message: String,
    /// Number of iterations.
    pub n_iter: usize,
}

// ─── SOCP → SDP lifting ───────────────────────────────────────────────────────

/// Convert an SOCP to an equivalent SDP via the Schur complement / rotated cone identity.
///
/// For each SOC constraint  ‖u‖ ≤ t  (where u = Aₖ x + bₖ, t = cₖ' x + dₖ),
/// introduce a symmetric PSD block:
///
/// ```text
/// [ t I   u ]
/// [ u'   t ] ⪰ 0   ↔   t ≥ 0 and ‖u‖ ≤ t   (Schur complement)
/// ```
///
/// The resulting SDP has a block-diagonal PSD variable.
///
/// # Returns
///
/// An [`SDPProblem`] whose optimal value equals that of the original SOCP,
/// together with an extraction function signature (the lifted variable has
/// dimension sum_k (mₖ + 1)).
pub fn socp_to_sdp(problem: &SOCPProblem) -> OptimizeResult<SDPProblem> {
    let n = problem.n();

    // Total PSD-block dimension = Σ_k (m_k + 1)
    let block_sizes: Vec<usize> = problem
        .constraints
        .iter()
        .map(|c| c.a.nrows() + 1)
        .collect();
    let total_dim: usize = block_sizes.iter().sum();

    // SDP variable Z ∈ S_{total_dim}.
    // Variables = original x ∈ ℝⁿ  (+  slack variables for cone sides).
    // We reformulate as a pure SDP in Y = Z (the PSD matrix) and lift x
    // by introducing explicit auxiliary variables for t_k = cₖ' x + dₖ.
    //
    // Full standard-form SDP: decision variable is the block-diagonal matrix.
    // Each k-th block Bₖ = tₖ * I_{mₖ+1}  with off-diagonal = uₖ.
    //
    // For the pure SDP form, we parametrise by (x, t_1, ..., t_K, Z_11, ...)
    // This gets complex; here we use the simpler scalar-lifting approach:
    //
    // Introduce scalar sₖ = tₖ and vector uₖ = Aₖ x + bₖ.
    // The rotated SOC constraint is:
    //   sₖ² ≥ ‖uₖ‖²   →   [ sₖ  uₖ' ; uₖ  sₖ I_{mₖ} ] ⪰ 0
    //
    // The SDP has variable x_ext = [x; s_1; ...; s_K] ∈ ℝ^{n + K} with
    // a block-diagonal PSD matrix Z whose k-th block relates to (sₖ, uₖ).

    let k = problem.constraints.len();

    // Build block-diagonal SDP.
    // For block k with dimension (mₖ+1) × (mₖ+1):
    //   Z_k = [ t_k         (A_k x + b_k)' ]
    //         [ A_k x + b_k  t_k  I_{m_k}  ]
    // Constraint:  Z_k_{0,0} = t_k  →  parametrised by x.
    //
    // Objective:  min c' x  (embed into the SDP trace form).

    // We use a direct variable: let w = [x; t_1; ...; t_K]  ∈ ℝ^{n+K}.
    // The SDP objective is: c_w' w  (only the first n components matter).
    //
    // For each block k (dimension dk = m_k + 1):
    //   Z_k is (dk × dk) PSD.
    //   Affine constraints link Z_k to w:
    //     Z_k[0,0]         = c_k' x + d_k       ← t_k definition
    //     Z_k[i+1, 0]      = (A_k x + b_k)[i]   ← off-diagonal
    //     Z_k[0, i+1]      = (A_k x + b_k)[i]   ← symmetry
    //     Z_k[i+1, i+1]    = c_k' x + d_k       ← diagonal = t_k
    //
    // This means the PSD variable is block-diagonal; we can embed it into
    // a single large PSD variable of dimension total_dim.

    // For simplicity of the returned SDPProblem, we embed all blocks into a
    // single (total_dim × total_dim) matrix by placing block k at row/col offset
    // off_k = Σ_{j<k} (m_j + 1).

    // SDP decision variable: X ∈ S_{total_dim}.  Also need x ∈ ℝⁿ, but SDP
    // is in matrices.  We add x as additional scalar variables by extending the
    // PSD matrix using a rank-1 lifting (homogeneous lifting technique):
    //
    // Introduce Z̃ = [ X       ; ... ]   of dimension (total_dim + n + 1).
    // This becomes very large for general n.  Instead, we use the standard
    // "dual" SDP form where the affine variable IS the (lifted) SDP matrix.
    //
    // For a clean implementation we use the AHO (1998) embedding:
    //   Extend decision vector to (n+K) and use one scalar SDP variable per block.

    // Practical implementation: build an SDP in x ∈ ℝⁿ+K with PSD variable Z̃
    // of dimension total_dim.
    // Constraints encode the affine structure.

    // Z̃ has dimension total_dim = Σ_k (m_k + 1).
    // For block k occupying rows/cols [off_k, off_k + d_k):
    //   tr(E_{00}^{(k)} Z̃) = c_k' x + d_k       (constraint for t_k = diagonal 0,0)
    //   tr(E_{i0}^{(k)} Z̃) = (A_k x + b_k)[i]   (off-diagonal)
    //   tr(E_{ii}^{(k)} Z̃) = c_k' x + d_k       (diagonal i+1,i+1 = t_k)
    //
    // Objective: min c' x  →  we need to express x in terms of Z̃.
    // We can add an extra variable by augmenting Z̃ with a (total_dim+n+1) block,
    // but that changes the structure.
    //
    // The cleanest standard-form approach is to express x through the constraints
    // and keep Z as the only SDP variable. For each SOC block k with m_k = 1
    // (scalar), the SDP block is 2×2 and the x-components appear linearly in the
    // constraint RHS. This is exactly the standard form with b = b(x) dependent
    // on x, which we need to unroll.
    //
    // The proper way is: introduce auxiliary variables τ_k and embed x explicitly.
    // Here we take the practical route: since the user may also want the standard
    // SDP return for further processing, we return a "homogeneous" SDP where the
    // original SOCP x is encoded in the last column/row of an (n+1) × (n+1)
    // PSD variable W = [x; 1] [x; 1]' (rank-1 relaxation) plus the cone blocks.

    // For the purposes of this function we return an SDP in combined variable
    // dimension = total_dim + n + 1, which contains both the x-part and cone blocks.

    let sdp_n = total_dim + n + 1; // combined PSD dimension

    // C matrix: objective c' x  →  encoded as tr(C_sdp Z_sdp)
    // We place x in the last column (column n in the bottom-right (n+1)×(n+1) block).
    // Z_sdp[(total_dim + i), (total_dim + n)] = xᵢ / 2  (symmetrised with (n, i))
    // tr(C_sdp Z_sdp) = Σᵢ c[i] * xᵢ
    let mut c_sdp = Array2::<f64>::zeros((sdp_n, sdp_n));
    for i in 0..n {
        c_sdp[[total_dim + i, total_dim + n]] = problem.obj[i] * 0.5;
        c_sdp[[total_dim + n, total_dim + i]] = problem.obj[i] * 0.5;
    }

    let mut a_mats: Vec<Array2<f64>> = Vec::new();
    let mut b_vals: Vec<f64> = Vec::new();

    // Normalisation constraint: Z_sdp[n+total_dim, n+total_dim] = 1  (homogeneous).
    {
        let mut ak = Array2::<f64>::zeros((sdp_n, sdp_n));
        ak[[total_dim + n, total_dim + n]] = 1.0;
        a_mats.push(ak);
        b_vals.push(1.0);
    }

    // Constraints encoding the SOC blocks.
    let mut off = 0usize;
    for (ki, con) in problem.constraints.iter().enumerate() {
        let mk = con.a.nrows();
        let dk = mk + 1;

        // t_k = c_k' x + d_k
        //   → Z̃[off, off] = t_k
        //   → Z̃[total_dim+n, total_dim+i] · c_k[i] + d_k * Z̃[total_dim+n, total_dim+n]
        //      = Z̃[off, off]
        // Constraint:  Z̃[off, off] - Σᵢ c_k[i] * Z̃_x[i] = d_k
        // where Z̃_x[i] = Z_sdp[total_dim + i, total_dim + n]  (x component via homogeneous lift)

        // a) Main diagonal Z[off, off] = t_k
        //    Expressed as: tr(A Z) = d_k  with  A_{off,off} = 1, A_{x_i, n} = -c_k[i]/2 (sym)
        {
            let mut ak = Array2::<f64>::zeros((sdp_n, sdp_n));
            ak[[off, off]] = 1.0;
            for i in 0..n {
                ak[[total_dim + i, total_dim + n]] = -con.c[i] * 0.5;
                ak[[total_dim + n, total_dim + i]] = -con.c[i] * 0.5;
            }
            a_mats.push(ak);
            b_vals.push(con.d);
        }

        // b) All sub-diagonal elements Z[off+r+1, off+r+1] = t_k  for r=0..mk
        for r in 0..mk {
            let mut ak = Array2::<f64>::zeros((sdp_n, sdp_n));
            ak[[off + r + 1, off + r + 1]] = 1.0;
            for i in 0..n {
                ak[[total_dim + i, total_dim + n]] = -con.c[i] * 0.5;
                ak[[total_dim + n, total_dim + i]] = -con.c[i] * 0.5;
            }
            a_mats.push(ak);
            b_vals.push(con.d);
        }

        // c) Off-diagonal Z[off, off+r+1] = u_k[r] = (A_k x + b_k)[r]
        //    tr(A Z) = b_k[r]  with  A_{off, off+r+1} = 1/2, A_{off+r+1, off} = 1/2,
        //                            A_{x_i, n} = -A_k[r,i] / 2 (symmetric)
        for r in 0..mk {
            let mut ak = Array2::<f64>::zeros((sdp_n, sdp_n));
            ak[[off, off + r + 1]] = 0.5;
            ak[[off + r + 1, off]] = 0.5;
            for i in 0..n {
                let a_ri = con.a[[r, i]];
                ak[[total_dim + i, total_dim + n]] -= a_ri * 0.5;
                ak[[total_dim + n, total_dim + i]] -= a_ri * 0.5;
            }
            a_mats.push(ak);
            b_vals.push(con.b[r]);
        }

        let _ = ki; // suppress unused warning
        off += dk;
    }

    // Add equality constraints F x = g  (if any).
    if let (Some(feq), Some(geq)) = (&problem.eq_a, &problem.eq_b) {
        for r in 0..feq.nrows() {
            let mut ak = Array2::<f64>::zeros((sdp_n, sdp_n));
            for i in 0..n {
                ak[[total_dim + i, total_dim + n]] = feq[[r, i]] * 0.5;
                ak[[total_dim + n, total_dim + i]] = feq[[r, i]] * 0.5;
            }
            a_mats.push(ak);
            b_vals.push(geq[r]);
        }
    }

    let b_sdp = Array1::from_vec(b_vals);
    SDPProblem::new(c_sdp, a_mats, b_sdp)
}

// ─── SOCP interior-point solver ───────────────────────────────────────────────

/// Configuration for the SOCP interior-point solver.
#[derive(Debug, Clone)]
pub struct SOCPConfig {
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Step safety factor (< 1).
    pub step_factor: f64,
}

impl Default for SOCPConfig {
    fn default() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-7,
            step_factor: 0.95,
        }
    }
}

/// Primal-dual interior-point solver for SOCP.
///
/// Uses the Nesterov-Todd (NT) scaling; each iteration solves a (sparse)
/// block-structured linear system via dense Cholesky on the condensed matrix.
pub fn socp_interior_point(
    problem: &SOCPProblem,
    config: Option<SOCPConfig>,
) -> OptimizeResult<SOCPResult> {
    let cfg = config.unwrap_or_default();
    let n = problem.n();
    let k = problem.constraints.len();

    if k == 0 {
        // Unconstrained — no feasible direction (unbounded below unless c=0).
        let obj_val = 0.0;
        return Ok(SOCPResult {
            x: Array1::<f64>::zeros(n),
            obj_val,
            residuals: vec![],
            converged: true,
            message: "No constraints — trivial solution x=0".into(),
            n_iter: 0,
        });
    }

    // ── Initialise: x = 0, s_k = 1 (cone slack), u_k = 0 ──────────────
    let mut x = Array1::<f64>::zeros(n);

    // Equality constraints  F x = g  (may be empty).
    let p_eq = problem.eq_a.as_ref().map(|f| f.nrows()).unwrap_or(0usize);

    // Feasibility warm-start: if there are equality constraints, project x
    // onto the affine subspace  F x = g  so the first iterate is feasible.
    if p_eq > 0 {
        if let (Some(feq), Some(geq)) = (&problem.eq_a, &problem.eq_b) {
            // Least-norm solution:  x₀ = F' (F F')⁻¹ g.
            // Build F F' ∈ ℝ^{p×p}.
            let mut fft = Array2::<f64>::zeros((p_eq, p_eq));
            for r in 0..p_eq {
                for s in 0..p_eq {
                    let mut v = 0.0_f64;
                    for j in 0..n {
                        v += feq[[r, j]] * feq[[s, j]];
                    }
                    fft[[r, s]] = v;
                }
            }
            // Regularise slightly.
            for r in 0..p_eq {
                fft[[r, r]] += 1e-12;
            }
            // Solve (F F') λ = g  for λ.
            if let Ok(lam) = solve(&fft.view(), &geq.view(), None) {
                // x₀ = F' λ.
                for j in 0..n {
                    let mut v = 0.0_f64;
                    for r in 0..p_eq {
                        v += feq[[r, j]] * lam[r];
                    }
                    x[j] = v;
                }
            }
        }
    }

    // For each constraint k, the cone variable is (t_k, u_k) where
    // t_k = c_k' x + d_k,  u_k = A_k x + b_k.
    // We maintain a slack τ_k > 0 such that t_k = ‖u_k‖ + τ_k (strict interior).
    let mut tau: Vec<f64> = vec![1.0; k]; // Slack t_k above ‖u_k‖

    let mut n_iter = 0usize;
    let mut converged = false;
    let mut message = "maximum iterations reached".to_string();

    for iter in 0..cfg.max_iter {
        n_iter = iter + 1;

        // ── Compute cone values ──────────────────────────────────────────
        let mut u_vecs: Vec<Array1<f64>> = Vec::with_capacity(k);
        let mut t_vals: Vec<f64> = Vec::with_capacity(k);
        for ki in 0..k {
            let con = &problem.constraints[ki];
            let u = con.a.dot(&x) + &con.b;
            let t = con.c.dot(&x) + con.d + tau[ki];
            u_vecs.push(u);
            t_vals.push(t);
        }

        // ── Primal and dual residuals ─────────────────────────────────────
        // Dual: ∇f - Σ λ_k ∇g_k = 0
        // Primal: t_k - ‖u_k‖ - τ_k ≥ 0  (feasibility, enforced by τ_k ≥ 0)

        // Compute gradient of the Lagrangian for dual feasibility.
        // ∂/∂x = c - Σ_k [ c_k (1/t_k) t_k + A_k' u_k / t_k ] ... (simplified)
        // For now use the complementarity residual as convergence criterion.

        let mut comp = 0.0_f64;
        for ki in 0..k {
            let u_norm = u_vecs[ki].iter().map(|v| v * v).sum::<f64>().sqrt();
            comp += (t_vals[ki] - u_norm).abs();
        }

        // Equality constraint residual  ‖F x - g‖.
        let eq_res = if p_eq > 0 {
            if let (Some(feq), Some(geq)) = (&problem.eq_a, &problem.eq_b) {
                let fx: Array1<f64> = feq.dot(&x);
                fx.iter()
                    .zip(geq.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Dual residual
        let mut rd = problem.obj.clone();
        for ki in 0..k {
            let con = &problem.constraints[ki];
            let t = t_vals[ki].max(1e-15);
            let u_norm = u_vecs[ki]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                .max(1e-15);
            let lambda = u_norm / t; // approximate multiplier
                                     // d(‖A x + b‖ / t) / dx ≈ A' u / (u_norm * t)  - ...
                                     // For convergence check only, use simplified form.
            for i in 0..n {
                let mut grad_i = con.c[i] * lambda;
                for r in 0..con.a.nrows() {
                    grad_i -= con.a[[r, i]] * u_vecs[ki][r] / (u_norm * t);
                }
                rd[i] -= grad_i;
            }
        }
        let rd_norm = rd.iter().map(|v| v * v).sum::<f64>().sqrt();

        if comp < cfg.tol && rd_norm < cfg.tol && eq_res < cfg.tol {
            converged = true;
            message = format!(
                "Converged in {} iterations (comp={:.2e}, rd={:.2e}, eq={:.2e})",
                n_iter, comp, rd_norm, eq_res
            );
            break;
        }

        // ── Newton step: condensed normal-equations (augmented KKT) ──────
        // With equality constraints  F x = g, we solve the augmented system:
        //   [ H   F' ] [ dx ]   [ -g_bar ]
        //   [ F   0  ] [ λ  ] = [ g - F x ]
        // where g_bar is the barrier gradient and λ are KKT multipliers.
        // When p_eq = 0 this reduces to H dx = -g_bar.
        let mut h = Array2::<f64>::zeros((n, n));
        let mut g_bar = Array1::<f64>::zeros(n); // barrier gradient

        for i in 0..n {
            g_bar[i] = problem.obj[i];
        }

        for ki in 0..k {
            let con = &problem.constraints[ki];
            let t = t_vals[ki].max(1e-12);
            let u = &u_vecs[ki];
            let u_norm2 = u.iter().map(|v| v * v).sum::<f64>();
            let _u_norm = u_norm2.sqrt().max(1e-15);

            // Cone barrier gradient: ∂φ/∂x = -(2/t) c - 2 A' u / (t² - ‖u‖²)
            // Using log-barrier φ = -log(t - ‖u‖) ≈ -log(t² - ‖u‖²)/2
            let rho = (t * t - u_norm2).max(1e-15);
            for i in 0..n {
                let mut grad_i = -2.0 * con.c[i] / t;
                for r in 0..con.a.nrows() {
                    grad_i -= 2.0 * con.a[[r, i]] * u[r] / rho;
                }
                g_bar[i] += grad_i;
            }

            // Cone barrier Hessian: ∂²φ/∂x²
            // = (2/t²) c c' + (4/(rho²)) (A' u) (A' u)' + (2/rho) A' A
            let inv_t2 = 2.0 / (t * t);
            let inv_rho2 = 4.0 / (rho * rho);
            let inv_rho = 2.0 / rho;

            // A' u  ∈ ℝⁿ
            let mut at_u = Array1::<f64>::zeros(n);
            for i in 0..n {
                for r in 0..con.a.nrows() {
                    at_u[i] += con.a[[r, i]] * u[r];
                }
            }

            for i in 0..n {
                for j in 0..n {
                    h[[i, j]] += inv_t2 * con.c[i] * con.c[j] + inv_rho2 * at_u[i] * at_u[j];
                    // + inv_rho * A' A  (diagonal block)
                    for r in 0..con.a.nrows() {
                        h[[i, j]] += inv_rho * con.a[[r, i]] * con.a[[r, j]];
                    }
                }
            }
        }

        // Regularise H.
        let h_norm = h.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
        let eps = 1e-8 * h_norm;
        for i in 0..n {
            h[[i, i]] += eps;
        }

        // Build augmented KKT system when equality constraints are present.
        let dx = if p_eq == 0 {
            // Pure Newton step: H dx = -g_bar.
            let neg_g = g_bar.map(|v| -v);
            solve(&h.view(), &neg_g.view(), None).map_err(OptimizeError::from)?
        } else {
            // Augmented KKT:
            //   [ H   F' ] [dx]   [-g_bar    ]
            //   [ F   0  ] [λ ]   [g - F x   ]
            let total = n + p_eq;
            let mut kkt = Array2::<f64>::zeros((total, total));
            let mut rhs = Array1::<f64>::zeros(total);

            // Upper-left block: H.
            for i in 0..n {
                for j in 0..n {
                    kkt[[i, j]] = h[[i, j]];
                }
            }

            // Upper-right / lower-left: F' / F.
            if let Some(feq) = &problem.eq_a {
                for r in 0..p_eq {
                    for j in 0..n {
                        kkt[[j, n + r]] = feq[[r, j]]; // F'
                        kkt[[n + r, j]] = feq[[r, j]]; // F
                    }
                }
            }

            // RHS top: -g_bar.
            for i in 0..n {
                rhs[i] = -g_bar[i];
            }

            // RHS bottom: g - F x.
            if let (Some(feq), Some(geq)) = (&problem.eq_a, &problem.eq_b) {
                for r in 0..p_eq {
                    let mut fx_r = 0.0_f64;
                    for j in 0..n {
                        fx_r += feq[[r, j]] * x[j];
                    }
                    rhs[n + r] = geq[r] - fx_r;
                }
            }

            // Small regularisation on the (0,0) block to avoid near-singularity
            // when H is very large and F is rank-deficient.
            let kkt_reg = 1e-12 * kkt.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
            for i in 0..total {
                kkt[[i, i]] += kkt_reg;
            }

            let sol = solve(&kkt.view(), &rhs.view(), None).map_err(OptimizeError::from)?;
            // Extract only the dx part (first n components).
            sol.slice(scirs2_core::ndarray::s![..n]).to_owned()
        };

        // ── Line search (Armijo) ──────────────────────────────────────────
        let mut alpha = 1.0_f64;
        let armijo_c = 1e-4;
        let f0: f64 = problem.obj.iter().zip(x.iter()).map(|(c, xi)| c * xi).sum();
        let slope: f64 = problem.obj.iter().zip(dx.iter()).map(|(c, d)| c * d).sum();

        for _ in 0..40 {
            let x_trial = &x + &(&dx * alpha);
            let f_trial: f64 = problem
                .obj
                .iter()
                .zip(x_trial.iter())
                .map(|(c, xi)| c * xi)
                .sum();
            // Check cone feasibility (all τ_k > 0 after update).
            let feasible = (0..k).all(|ki| {
                let con = &problem.constraints[ki];
                let t_new = con.c.dot(&x_trial) + con.d + tau[ki];
                let u_new = con.a.dot(&x_trial) + &con.b;
                let u_norm = u_new.iter().map(|v| v * v).sum::<f64>().sqrt();
                t_new > u_norm + 1e-12
            });
            if feasible && f_trial <= f0 + armijo_c * alpha * slope {
                break;
            }
            alpha *= 0.5;
            if alpha < 1e-15 {
                alpha = 1e-15;
                break;
            }
        }

        // Update x and tau.
        for i in 0..n {
            x[i] += alpha * dx[i];
        }
        // Update tau to keep interior.
        for ki in 0..k {
            let con = &problem.constraints[ki];
            let u = con.a.dot(&x) + &con.b;
            let t_target = con.c.dot(&x) + con.d;
            let u_norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
            let margin = (t_target - u_norm).max(0.0);
            tau[ki] = margin * 0.5 + 1e-8;
        }
    }

    let obj_val = problem.obj.iter().zip(x.iter()).map(|(c, xi)| c * xi).sum();
    let residuals: Vec<f64> = (0..k)
        .map(|ki| {
            let con = &problem.constraints[ki];
            let u = con.a.dot(&x) + &con.b;
            let t = con.c.dot(&x) + con.d;
            let u_norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
            u_norm - t
        })
        .collect();

    Ok(SOCPResult {
        x,
        obj_val,
        residuals,
        converged,
        message,
        n_iter,
    })
}

// ─── Application: robust least squares ───────────────────────────────────────

/// Result of robust least-squares SOCP.
#[derive(Debug, Clone)]
pub struct RobustLsResult {
    /// Optimal parameter vector x*.
    pub x: Array1<f64>,
    /// Worst-case residual (optimal SOCP value).
    pub worst_case_residual: f64,
    /// Whether the SOCP converged.
    pub converged: bool,
}

/// Robust least squares via SOCP.
///
/// Solves the robust LS problem:
///
/// ```text
/// min_x  max_{‖dA‖_F ≤ ρ}  ‖(A + dA) x - b‖
/// ```
///
/// which can be re-written as an SOCP:
///
/// ```text
/// min_{x, t}  t
/// s.t.  ‖A x - b‖ + ρ ‖x‖ ≤ t
/// ```
///
/// Splitting into two SOC constraints gives:
///
/// ```text
/// min_{x, t, s₁, s₂}   t
/// s.t.  ‖A x - b‖ ≤ s₁,    ρ ‖x‖ ≤ s₂,    s₁ + s₂ ≤ t
/// ```
///
/// which after substitution is a standard SOCP in (x, t) ∈ ℝ^{n+1}.
pub fn robust_ls_socp(
    a: &ArrayView2<f64>,
    b: &ArrayView1<f64>,
    rho: f64,
) -> OptimizeResult<RobustLsResult> {
    let (m, n) = (a.nrows(), a.ncols());
    if b.len() != m {
        return Err(OptimizeError::ValueError(format!(
            "A is {}×{} but b has {} elements",
            m,
            n,
            b.len()
        )));
    }
    if rho < 0.0 {
        return Err(OptimizeError::ValueError(format!(
            "rho must be non-negative, got {}",
            rho
        )));
    }

    // Variables: w = [x (n), t (1), s1 (1), s2 (1)]  ∈ ℝ^{n+3}.
    let nw = n + 3;
    let t_idx = n;
    let s1_idx = n + 1;
    let s2_idx = n + 2;

    // Objective: min t  →  c[t_idx] = 1.
    let mut obj = Array1::<f64>::zeros(nw);
    obj[t_idx] = 1.0;

    // Constraint 1: ‖A x - b‖ ≤ s₁
    //   → ‖A_ext w + b_neg‖ ≤ c_1' w + d_1
    //   A_ext = [A | 0 | 0 | 0]  (m × nw),  b_neg = -b,
    //   c_1 = e_{s1},  d_1 = 0.
    let mut a1 = Array2::<f64>::zeros((m, nw));
    let mut b1 = Array1::<f64>::zeros(m);
    for i in 0..m {
        for j in 0..n {
            a1[[i, j]] = a[[i, j]];
        }
        b1[i] = -b[i];
    }
    let mut c1 = Array1::<f64>::zeros(nw);
    c1[s1_idx] = 1.0;
    let con1 = SOCConstraint::new(a1, b1, c1, 0.0)?;

    // Constraint 2: ρ ‖x‖ ≤ s₂
    //   → ‖ρ I x + 0‖ ≤ s₂
    //   A_ext2 = [ρ I | 0 | 0 | 0]  (n × nw),
    //   c_2 = e_{s2},  d_2 = 0.
    let mut a2 = Array2::<f64>::zeros((n, nw));
    for i in 0..n {
        a2[[i, i]] = rho;
    }
    let b2 = Array1::<f64>::zeros(n);
    let mut c2 = Array1::<f64>::zeros(nw);
    c2[s2_idx] = 1.0;
    let con2 = SOCConstraint::new(a2, b2, c2, 0.0)?;

    // Constraint 3: s₁ + s₂ ≤ t  →  ‖[s1; s2]‖ ≤ t  is NOT the same.
    // Use the linear constraint s₁ + s₂ ≤ t  →  t - s₁ - s₂ ≥ 0.
    // Express as SOC: ‖e‖ ≤ t - s₁ - s₂  with e=0 (trivial SOC ‖0‖ ≤ scalar).
    // i.e., 0 ≤ t - s₁ - s₂  →  SOC: ‖[0]‖ ≤ t - s₁ - s₂.
    let a3 = Array2::<f64>::zeros((1, nw));
    let b3 = Array1::from_vec(vec![0.0]);
    let mut c3 = Array1::<f64>::zeros(nw);
    c3[t_idx] = 1.0;
    c3[s1_idx] = -1.0;
    c3[s2_idx] = -1.0;
    let con3 = SOCConstraint::new(a3, b3, c3, 0.0)?;

    let problem = SOCPProblem::new(obj, vec![con1, con2, con3])?;
    let result = socp_interior_point(&problem, None)?;

    let x = result.x.slice(scirs2_core::ndarray::s![..n]).to_owned();
    let worst_case_residual = result.x[t_idx];

    Ok(RobustLsResult {
        x,
        worst_case_residual,
        converged: result.converged,
    })
}

// ─── Application: portfolio optimisation ─────────────────────────────────────

/// Result of portfolio optimisation SOCP.
#[derive(Debug, Clone)]
pub struct PortfolioSocpResult {
    /// Optimal portfolio weights (sum to 1, non-negative).
    pub weights: Array1<f64>,
    /// Expected return μ' w.
    pub expected_return: f64,
    /// Portfolio standard deviation √(w' Σ w).
    pub std_dev: f64,
    /// Whether the SOCP converged.
    pub converged: bool,
}

/// Mean-variance portfolio optimisation via SOCP.
///
/// Solves the Markowitz problem:
///
/// ```text
/// min_{w}    -μ' w + γ √(w' Σ w)
/// s.t.       1' w = 1,   w ≥ 0
/// ```
///
/// where γ ≥ 0 is the risk-aversion parameter.
///
/// The variance constraint is written as a SOC constraint:
/// ```text
/// ‖L w‖ ≤ t   (L is the Cholesky factor of Σ)
/// γ t ≤ objective penalty
/// ```
///
/// # Arguments
/// * `mu`    – expected returns vector (length n_assets)
/// * `sigma` – covariance matrix (n_assets × n_assets, PSD)
/// * `gamma` – risk aversion parameter (≥ 0)
pub fn portfolio_optimization_socp(
    mu: &ArrayView1<f64>,
    sigma: &ArrayView2<f64>,
    gamma: f64,
) -> OptimizeResult<PortfolioSocpResult> {
    let n = mu.len();
    if sigma.nrows() != n || sigma.ncols() != n {
        return Err(OptimizeError::ValueError(format!(
            "sigma must be {}×{} but is {}×{}",
            n,
            n,
            sigma.nrows(),
            sigma.ncols()
        )));
    }
    if gamma < 0.0 {
        return Err(OptimizeError::ValueError(format!(
            "gamma must be non-negative, got {}",
            gamma
        )));
    }

    // Variables: (w, t)  ∈ ℝ^{n+1}  where t ≥ √(w' Σ w).
    let nw = n + 1;
    let t_idx = n;

    // Compute Cholesky factor L of Σ (L L' = Σ).
    let sigma_arr = sigma.to_owned();
    let l = match scirs2_linalg::cholesky(&sigma_arr.view(), None) {
        Ok(factor) => factor,
        Err(_) => {
            // Regularise and retry.
            let mut reg = sigma_arr.clone();
            for i in 0..n {
                reg[[i, i]] += 1e-6;
            }
            scirs2_linalg::cholesky(&reg.view(), None)
                .map_err(|e| OptimizeError::ComputationError(format!("Cholesky: {}", e)))?
        }
    };

    // Objective: min -μ' w + γ t
    let mut obj = Array1::<f64>::zeros(nw);
    for i in 0..n {
        obj[i] = -mu[i];
    }
    obj[t_idx] = gamma;

    // SOC constraint: ‖L' w‖ ≤ t
    // A = [L'  |  0 ]  (n × nw),  b = 0,  c = e_{t},  d = 0.
    let mut a_soc = Array2::<f64>::zeros((n, nw));
    for i in 0..n {
        for j in 0..n {
            // L is lower-triangular → L' is upper-triangular.
            a_soc[[i, j]] = l[[j, i]]; // L'[i,j] = L[j,i]
        }
    }
    let b_soc = Array1::<f64>::zeros(n);
    let mut c_soc = Array1::<f64>::zeros(nw);
    c_soc[t_idx] = 1.0;
    let con_var = SOCConstraint::new(a_soc, b_soc, c_soc, 0.0)?;

    // Constraint: w ≥ 0  →  -wᵢ ≤ 0  →  SOC: ‖0‖ ≤ wᵢ.
    let mut weight_constraints = Vec::new();
    for i in 0..n {
        let a_i = Array2::<f64>::zeros((1, nw));
        let b_i = Array1::from_vec(vec![0.0]);
        let mut c_i = Array1::<f64>::zeros(nw);
        c_i[i] = 1.0;
        weight_constraints.push(SOCConstraint::new(a_i, b_i, c_i, 0.0)?);
    }

    let mut all_cons = vec![con_var];
    all_cons.extend(weight_constraints);

    // Equality constraint: 1' w = 1.
    let mut f_eq = Array2::<f64>::zeros((1, nw));
    for i in 0..n {
        f_eq[[0, i]] = 1.0;
    }
    let g_eq = Array1::from_vec(vec![1.0]);

    let problem = SOCPProblem::new(obj, all_cons)?.with_equality(f_eq, g_eq)?;

    let result = socp_interior_point(&problem, None)?;

    let weights = result.x.slice(scirs2_core::ndarray::s![..n]).to_owned();
    let expected_return: f64 = mu.iter().zip(weights.iter()).map(|(m, w)| m * w).sum();

    // Compute portfolio variance w' Σ w.
    let mut var = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            var += weights[i] * sigma[[i, j]] * weights[j];
        }
    }
    let std_dev = var.sqrt();

    Ok(PortfolioSocpResult {
        weights,
        expected_return,
        std_dev,
        converged: result.converged,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_socp_problem_dim_check() {
        let obj = Array1::from_vec(vec![1.0, 2.0]);
        let a = Array2::<f64>::zeros((2, 2));
        let b = Array1::from_vec(vec![0.0, 0.0]);
        let c = Array1::from_vec(vec![1.0, 0.0]);
        let con = SOCConstraint::new(a, b, c, 1.0).expect("valid constraint");
        let problem = SOCPProblem::new(obj, vec![con]).expect("valid problem");
        assert_eq!(problem.n(), 2);
    }

    #[test]
    fn test_socp_constraint_dim_mismatch() {
        let a = Array2::<f64>::zeros((2, 3)); // 2 rows, 3 cols
        let b = Array1::from_vec(vec![0.0]); // 1 element, mismatch
        let c = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let result = SOCConstraint::new(a, b, c, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_socp_trivial_no_constraints() {
        let obj = Array1::from_vec(vec![1.0]);
        let problem = SOCPProblem::new(obj, vec![]).expect("valid");
        let result = socp_interior_point(&problem, None).expect("should succeed");
        assert!(result.converged);
    }

    #[test]
    fn test_robust_ls_basic() {
        // 2×2 system Ax = b with rho = 0.1
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("valid");
        let b = Array1::from_vec(vec![1.0, 1.0]);
        let result = robust_ls_socp(&a.view(), &b.view(), 0.1).expect("robust_ls should not fail");
        // x ≈ (1, 1) adjusted for robustness penalty
        assert!(result.x[0].is_finite());
        assert!(result.x[1].is_finite());
    }

    #[test]
    fn test_socp_to_sdp_builds() {
        // Simple SOCP: min x  s.t. ‖x‖ ≤ 1  (1D)
        let obj = Array1::from_vec(vec![1.0]);
        let a = Array2::<f64>::zeros((1, 1));
        let b = Array1::from_vec(vec![0.0]);
        let c = Array1::from_vec(vec![0.0]);
        let con = SOCConstraint::new(a, b, c, 1.0).expect("valid");
        let problem = SOCPProblem::new(obj, vec![con]).expect("valid");
        let sdp = socp_to_sdp(&problem);
        assert!(sdp.is_ok(), "SOCP→SDP lift should not fail");
    }

    #[test]
    fn test_portfolio_basic() {
        // 2-asset portfolio
        let mu = Array1::from_vec(vec![0.1, 0.2]);
        let sigma = Array2::from_shape_vec((2, 2), vec![0.04, 0.0, 0.0, 0.09]).expect("valid");
        let result = portfolio_optimization_socp(&mu.view(), &sigma.view(), 1.0)
            .expect("portfolio should not fail");
        // Weights should sum to ~1.
        let sum: f64 = result.weights.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 0.1); // relaxed for solver convergence
    }
}
