//! Advanced Fredholm integral equation solvers
//!
//! This module implements advanced methods for Fredholm integral equations of the second kind:
//! ```text
//! u(x) = f(x) + λ ∫ₐᵇ K(x, t) u(t) dt,   x ∈ [a, b]
//! ```
//!
//! ## Methods
//!
//! - **Nyström method**: Gaussian quadrature discretization → linear system
//! - **Degenerate kernel method**: Exploits separable structure K(x,t) = Σ aᵢ(x)bᵢ(t)
//! - **Neumann series**: Successive approximations for small |λ|
//!
//! ## References
//!
//! - Atkinson (1997), *The Numerical Solution of Integral Equations of the Second Kind*
//! - Delves & Mohamed (1985), *Computational Methods for Integral Equations*
//! - Kress (1999), *Linear Integral Equations*, 2nd ed.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};

// ──────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────

/// Gauss-Legendre nodes and weights on \[-1, 1\] up to 12 points.
pub(crate) fn gauss_legendre_nodes(npts: usize) -> (Vec<f64>, Vec<f64>) {
    match npts {
        1 => (vec![0.0], vec![2.0]),
        2 => (
            vec![-0.577_350_269_189_625_8, 0.577_350_269_189_625_8],
            vec![1.0, 1.0],
        ),
        3 => (
            vec![-0.774_596_669_241_483_4, 0.0, 0.774_596_669_241_483_4],
            vec![
                0.555_555_555_555_556,
                0.888_888_888_888_889,
                0.555_555_555_555_556,
            ],
        ),
        4 => (
            vec![
                -0.861_136_311_594_052_6,
                -0.339_981_043_584_856_3,
                0.339_981_043_584_856_3,
                0.861_136_311_594_052_6,
            ],
            vec![
                0.347_854_845_137_453_8,
                0.652_145_154_862_546_1,
                0.652_145_154_862_546_1,
                0.347_854_845_137_453_8,
            ],
        ),
        5 => (
            vec![
                -0.906_179_845_938_664,
                -0.538_469_310_105_683_1,
                0.0,
                0.538_469_310_105_683_1,
                0.906_179_845_938_664,
            ],
            vec![
                0.236_926_885_056_189_1,
                0.478_628_670_499_366_5,
                0.568_888_888_888_889,
                0.478_628_670_499_366_5,
                0.236_926_885_056_189_1,
            ],
        ),
        8 => (
            vec![
                -0.960_289_856_497_536_3,
                -0.796_666_477_413_626_7,
                -0.525_532_409_916_329,
                -0.183_434_642_495_649_8,
                0.183_434_642_495_649_8,
                0.525_532_409_916_329,
                0.796_666_477_413_626_7,
                0.960_289_856_497_536_3,
            ],
            vec![
                0.101_228_536_290_376_3,
                0.222_381_034_453_374_5,
                0.313_706_645_877_887_3,
                0.362_683_783_378_362,
                0.362_683_783_378_362,
                0.313_706_645_877_887_3,
                0.222_381_034_453_374_5,
                0.101_228_536_290_376_3,
            ],
        ),
        10 => (
            vec![
                -0.973_906_528_517_171_7,
                -0.865_063_366_688_984_5,
                -0.679_409_568_299_024_4,
                -0.433_395_394_129_247_2,
                -0.148_874_338_981_631_2,
                0.148_874_338_981_631_2,
                0.433_395_394_129_247_2,
                0.679_409_568_299_024_4,
                0.865_063_366_688_984_5,
                0.973_906_528_517_171_7,
            ],
            vec![
                0.066_671_344_086_681_0,
                0.149_451_349_150_580_6,
                0.219_086_362_515_982,
                0.269_266_719_309_996_4,
                0.295_524_224_714_752_9,
                0.295_524_224_714_752_9,
                0.269_266_719_309_996_4,
                0.219_086_362_515_982,
                0.149_451_349_150_580_6,
                0.066_671_344_086_681_0,
            ],
        ),
        12 => (
            vec![
                -0.981_560_634_246_719_2,
                -0.904_117_256_370_474_9,
                -0.769_902_674_194_304_7,
                -0.587_317_954_286_617_7,
                -0.367_831_498_998_180_2,
                -0.125_233_408_511_468_9,
                0.125_233_408_511_468_9,
                0.367_831_498_998_180_2,
                0.587_317_954_286_617_7,
                0.769_902_674_194_304_7,
                0.904_117_256_370_474_9,
                0.981_560_634_246_719_2,
            ],
            vec![
                0.047_175_336_386_511_8,
                0.106_939_325_995_318_4,
                0.160_078_328_543_346_2,
                0.203_167_426_723_065_9,
                0.233_492_536_538_354_8,
                0.249_147_045_813_402_8,
                0.249_147_045_813_402_8,
                0.233_492_536_538_354_8,
                0.203_167_426_723_065_9,
                0.160_078_328_543_346_2,
                0.106_939_325_995_318_4,
                0.047_175_336_386_511_8,
            ],
        ),
        _ => {
            // Fall back to 5-point rule
            gauss_legendre_nodes(5)
        }
    }
}

/// Gaussian elimination with partial pivoting solving A x = b in place.
fn gauss_solve(a: &mut Array2<f64>, b: &mut Array1<f64>) -> IntegrateResult<Array1<f64>> {
    let n = b.len();
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[[col, col]].abs();
        for row in (col + 1)..n {
            let v = a[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return Err(IntegrateError::LinearSolveError(
                "Singular or near-singular Nyström matrix".to_string(),
            ));
        }
        if max_row != col {
            for j in col..n {
                let tmp = a[[col, j]];
                a[[col, j]] = a[[max_row, j]];
                a[[max_row, j]] = tmp;
            }
            b.swap(col, max_row);
        }
        let pivot = a[[col, col]];
        for row in (col + 1)..n {
            let factor = a[[row, col]] / pivot;
            for j in col..n {
                let sub = factor * a[[col, j]];
                a[[row, j]] -= sub;
            }
            let sub_b = factor * b[col];
            b[row] -= sub_b;
        }
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[[i, j]] * x[j];
        }
        x[i] = s / a[[i, i]];
    }
    Ok(x)
}

// ──────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────

/// Configuration for the Fredholm second-kind solver.
#[derive(Debug, Clone)]
pub struct FredholmSolverConfig {
    /// Left endpoint of the integration interval.
    pub a: f64,
    /// Right endpoint of the integration interval.
    pub b: f64,
    /// Number of Gauss-Legendre quadrature points (2–12 supported; default 8).
    pub n_quad: usize,
    /// Multiplier λ in front of the integral.
    pub lambda: f64,
    /// Maximum Neumann-series terms (only used by `fredholm_neumann`).
    pub neumann_max_terms: usize,
    /// Convergence tolerance for the Neumann series.
    pub neumann_tol: f64,
}

impl Default for FredholmSolverConfig {
    fn default() -> Self {
        Self {
            a: 0.0,
            b: 1.0,
            n_quad: 8,
            lambda: 1.0,
            neumann_max_terms: 50,
            neumann_tol: 1e-12,
        }
    }
}

/// Solution returned by Fredholm solvers.
#[derive(Debug, Clone)]
pub struct FredholmSolution {
    /// Collocation / quadrature nodes xᵢ.
    pub nodes: Vec<f64>,
    /// Quadrature weights wᵢ (useful for post-processing).
    pub weights: Vec<f64>,
    /// Solution u(xᵢ) at the collocation nodes.
    pub u: Vec<f64>,
    /// Estimated condition number of the Nyström matrix (Nyström method only).
    pub condition_estimate: f64,
    /// Number of Neumann-series terms used (Neumann method only).
    pub n_terms_used: usize,
}

// ──────────────────────────────────────────────────────────────────────────
// FredholmSolver
// ──────────────────────────────────────────────────────────────────────────

/// High-level solver for Fredholm integral equations of the second kind:
/// ```text
/// u(x) = f(x) + λ ∫ₐᵇ K(x, t) u(t) dt
/// ```
///
/// Contains three method dispatchers:
/// - [`FredholmSolver::nystrom_method`] — Nyström Gaussian-quadrature system
/// - [`FredholmSolver::degenerate_kernel`] — analytic reduction for separable kernels
/// - [`FredholmSolver::neumann_series`] — successive approximations
pub struct FredholmSolver {
    /// Kernel function K(x, t).
    pub kernel: Box<dyn Fn(f64, f64) -> f64>,
    /// Configuration parameters.
    pub config: FredholmSolverConfig,
}

impl FredholmSolver {
    /// Create a new solver with the given kernel and config.
    pub fn new(kernel: impl Fn(f64, f64) -> f64 + 'static, config: FredholmSolverConfig) -> Self {
        Self {
            kernel: Box::new(kernel),
            config,
        }
    }

    /// Nyström method: discretise the integral with *n* Gauss-Legendre points,
    /// then solve the resulting n×n linear system (I − λ W K) **u** = **f**.
    ///
    /// After solving, the solution at arbitrary x can be recovered via the
    /// Nyström interpolation formula
    /// ```text
    /// u(x) = f(x) + λ Σⱼ wⱼ K(x, tⱼ) u(tⱼ)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `f` — right-hand side function f(x)
    ///
    /// # Errors
    ///
    /// Returns [`IntegrateError::LinearSolveError`] if the system is singular.
    pub fn nystrom_method(&self, f: impl Fn(f64) -> f64) -> IntegrateResult<FredholmSolution> {
        nystrom_method(&self.kernel, &f, &self.config)
    }

    /// Degenerate-kernel (separable-kernel) method.
    ///
    /// For a kernel of the form K(x, t) = Σᵢ aᵢ(x) bᵢ(t), the equation
    /// reduces to a finite-dimensional system of *rank* equations in the
    /// constants cᵢ = ∫ bᵢ(t) u(t) dt.
    ///
    /// The solution is then u(x) = f(x) + λ Σᵢ cᵢ aᵢ(x).
    ///
    /// # Arguments
    ///
    /// * `f` — right-hand side
    /// * `basis_a` — slice of *a*-basis functions \[a₁, a₂, …, aᵣ\]
    /// * `basis_b` — slice of *b*-basis functions \[b₁, b₂, …, bᵣ\] (same length)
    /// * `eval_points` — points at which to return the solution
    ///
    /// # Errors
    ///
    /// Returns [`IntegrateError::LinearSolveError`] if the reduced system is singular,
    /// or [`IntegrateError::InvalidInput`] for mismatched basis lengths.
    pub fn degenerate_kernel(
        &self,
        f: impl Fn(f64) -> f64,
        basis_a: &[Box<dyn Fn(f64) -> f64>],
        basis_b: &[Box<dyn Fn(f64) -> f64>],
        eval_points: &[f64],
    ) -> IntegrateResult<FredholmSolution> {
        degenerate_kernel(&self.config, &f, basis_a, basis_b, eval_points)
    }

    /// Neumann series (successive approximations):
    /// ```text
    /// u₀(x) = f(x)
    /// uₙ₊₁(x) = f(x) + λ ∫ K(x,t) uₙ(t) dt
    /// ```
    ///
    /// Converges when |λ| ‖K‖ < 1 (in the appropriate norm).
    ///
    /// # Arguments
    ///
    /// * `f` — right-hand side evaluated on the internal quadrature grid
    ///
    /// # Errors
    ///
    /// Returns [`IntegrateError::ConvergenceError`] if the series does not converge
    /// within `config.neumann_max_terms` iterations.
    pub fn neumann_series(&self, f: impl Fn(f64) -> f64) -> IntegrateResult<FredholmSolution> {
        fredholm_neumann(&self.kernel, &f, &self.config)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// nystrom_method (standalone)
// ──────────────────────────────────────────────────────────────────────────

/// Solve a Fredholm second-kind equation using the Nyström method.
///
/// This is also exposed as a standalone function for cases where the caller
/// does not wish to create a [`FredholmSolver`].
///
/// # Arguments
///
/// * `kernel` — kernel K(x, t)
/// * `f` — right-hand side f(x)
/// * `cfg` — solver configuration
///
/// # Returns
///
/// [`FredholmSolution`] with `nodes`, `weights`, and `u` values.
pub fn nystrom_method(
    kernel: &dyn Fn(f64, f64) -> f64,
    f: &dyn Fn(f64) -> f64,
    cfg: &FredholmSolverConfig,
) -> IntegrateResult<FredholmSolution> {
    let n = cfg.n_quad;
    if n == 0 {
        return Err(IntegrateError::InvalidInput(
            "n_quad must be at least 1".to_string(),
        ));
    }

    let a = cfg.a;
    let b = cfg.b;
    let lam = cfg.lambda;

    // GL nodes/weights on [-1,1] → map to [a,b]
    let (xi_ref, wi_ref) = gauss_legendre_nodes(n);
    let half_len = (b - a) * 0.5;
    let mid = (a + b) * 0.5;
    let nodes: Vec<f64> = xi_ref.iter().map(|&xi| mid + half_len * xi).collect();
    let weights: Vec<f64> = wi_ref.iter().map(|&wi| wi * half_len).collect();

    // Build system (I − λ W K) u = f
    let mut a_mat = Array2::<f64>::zeros((n, n));
    let mut rhs = Array1::<f64>::zeros(n);

    for i in 0..n {
        rhs[i] = f(nodes[i]);
        for j in 0..n {
            let k_ij = kernel(nodes[i], nodes[j]);
            a_mat[[i, j]] = if i == j {
                1.0 - lam * weights[j] * k_ij
            } else {
                -lam * weights[j] * k_ij
            };
        }
    }

    // Condition estimate: diagonal ratio
    let diag_max = (0..n).fold(f64::NEG_INFINITY, |m, i| m.max(a_mat[[i, i]].abs()));
    let diag_min = (0..n).fold(f64::INFINITY, |m, i| m.min(a_mat[[i, i]].abs()));
    let condition_estimate = if diag_min > 1e-300 {
        diag_max / diag_min
    } else {
        f64::INFINITY
    };

    let u_vec = gauss_solve(&mut a_mat, &mut rhs)?;

    Ok(FredholmSolution {
        nodes,
        weights,
        u: u_vec.to_vec(),
        condition_estimate,
        n_terms_used: 0,
    })
}

/// Evaluate the Nyström interpolant at an arbitrary point `x`.
///
/// Uses the formula u(x) = f(x) + λ Σⱼ wⱼ K(x, tⱼ) u(tⱼ).
pub fn nystrom_evaluate(
    x: f64,
    kernel: &dyn Fn(f64, f64) -> f64,
    f: &dyn Fn(f64) -> f64,
    sol: &FredholmSolution,
    lambda: f64,
) -> f64 {
    let sum: f64 = sol
        .nodes
        .iter()
        .zip(sol.weights.iter())
        .zip(sol.u.iter())
        .map(|((&tj, &wj), &uj)| wj * kernel(x, tj) * uj)
        .sum();
    f(x) + lambda * sum
}

// ──────────────────────────────────────────────────────────────────────────
// degenerate_kernel (standalone)
// ──────────────────────────────────────────────────────────────────────────

/// Solve a Fredholm second-kind equation with a **degenerate (separable) kernel**:
/// ```text
/// K(x, t) = Σᵣ aᵣ(x) bᵣ(t)
/// ```
///
/// The integral equation reduces to a finite-dimensional system.  Let
/// ```text
/// cᵣ = ∫ₐᵇ bᵣ(t) u(t) dt
/// ```
///
/// Substituting u(t) = f(t) + λ Σₛ cₛ aₛ(t) and integrating gives the
/// linear system
/// ```text
/// Σₛ (δᵣₛ − λ Bᵣₛ) cₛ = dᵣ
/// where  Bᵣₛ = ∫ₐᵇ bᵣ(t) aₛ(t) dt
///        dᵣ  = ∫ₐᵇ bᵣ(t) f(t) dt
/// ```
///
/// Once **c** is found, u(x) = f(x) + λ Σᵣ cᵣ aᵣ(x).
///
/// # Arguments
///
/// * `cfg` — solver configuration (interval, λ, quadrature points for numerical integration)
/// * `f` — right-hand side function
/// * `basis_a` — a-basis functions \[a₁, …, aᵣ\]
/// * `basis_b` — b-basis functions \[b₁, …, bᵣ\] (same length as `basis_a`)
/// * `eval_points` — points at which to return u(x)
///
/// # Errors
///
/// Returns [`IntegrateError::InvalidInput`] for empty or mismatched bases,
/// or [`IntegrateError::LinearSolveError`] if the reduced system is singular.
pub fn degenerate_kernel(
    cfg: &FredholmSolverConfig,
    f: &dyn Fn(f64) -> f64,
    basis_a: &[Box<dyn Fn(f64) -> f64>],
    basis_b: &[Box<dyn Fn(f64) -> f64>],
    eval_points: &[f64],
) -> IntegrateResult<FredholmSolution> {
    let rank = basis_a.len();
    if rank == 0 {
        return Err(IntegrateError::InvalidInput(
            "basis_a must be non-empty".to_string(),
        ));
    }
    if basis_b.len() != rank {
        return Err(IntegrateError::InvalidInput(format!(
            "basis_a (len={}) and basis_b (len={}) must have the same length",
            rank,
            basis_b.len()
        )));
    }
    if eval_points.is_empty() {
        return Err(IntegrateError::InvalidInput(
            "eval_points must be non-empty".to_string(),
        ));
    }

    let a = cfg.a;
    let b = cfg.b;
    let lam = cfg.lambda;
    let n_quad = cfg.n_quad.max(rank * 4); // ensure enough quadrature points

    // Build Gauss-Legendre quadrature on [a, b]
    let (xi_ref, wi_ref) = gauss_legendre_nodes(n_quad.min(12));
    let half_len = (b - a) * 0.5;
    let mid = (a + b) * 0.5;
    let nodes: Vec<f64> = xi_ref.iter().map(|&xi| mid + half_len * xi).collect();
    let weights: Vec<f64> = wi_ref.iter().map(|&wi| wi * half_len).collect();
    let nq = nodes.len();

    // Compute Bᵣₛ = ∫ bᵣ(t) aₛ(t) dt ≈ Σₖ wₖ bᵣ(tₖ) aₛ(tₖ)
    let mut b_mat = Array2::<f64>::zeros((rank, rank));
    for r in 0..rank {
        for s in 0..rank {
            let val: f64 = (0..nq)
                .map(|k| weights[k] * basis_b[r](nodes[k]) * basis_a[s](nodes[k]))
                .sum();
            b_mat[[r, s]] = val;
        }
    }

    // Compute dᵣ = ∫ bᵣ(t) f(t) dt ≈ Σₖ wₖ bᵣ(tₖ) f(tₖ)
    let mut d_vec = Array1::<f64>::zeros(rank);
    for r in 0..rank {
        d_vec[r] = (0..nq)
            .map(|k| weights[k] * basis_b[r](nodes[k]) * f(nodes[k]))
            .sum();
    }

    // Build (I − λ B) c = d
    let mut sys = Array2::<f64>::zeros((rank, rank));
    for r in 0..rank {
        for s in 0..rank {
            sys[[r, s]] = if r == s {
                1.0 - lam * b_mat[[r, s]]
            } else {
                -lam * b_mat[[r, s]]
            };
        }
    }

    let c_vec = gauss_solve(&mut sys, &mut d_vec)?;

    // Evaluate u(x) = f(x) + λ Σᵣ cᵣ aᵣ(x) at each eval_point
    let u_vals: Vec<f64> = eval_points
        .iter()
        .map(|&x| {
            let correction: f64 = (0..rank).map(|r| c_vec[r] * basis_a[r](x)).sum();
            f(x) + lam * correction
        })
        .collect();

    // Condition estimate for the reduced matrix
    let diag_max = (0..rank).fold(f64::NEG_INFINITY, |m, i| m.max(sys[[i, i]].abs()));
    let diag_min = (0..rank).fold(f64::INFINITY, |m, i| m.min(sys[[i, i]].abs()));
    let condition_estimate = if diag_min > 1e-300 {
        diag_max / diag_min
    } else {
        f64::INFINITY
    };

    Ok(FredholmSolution {
        nodes: eval_points.to_vec(),
        weights: vec![1.0; eval_points.len()],
        u: u_vals,
        condition_estimate,
        n_terms_used: 0,
    })
}

// ──────────────────────────────────────────────────────────────────────────
// fredholm_neumann (standalone)
// ──────────────────────────────────────────────────────────────────────────

/// Solve a Fredholm second-kind equation via the **Neumann series**
/// (successive approximations):
/// ```text
/// u₀ = f
/// uₙ₊₁(x) = f(x) + λ ∫ K(x, t) uₙ(t) dt
/// ```
///
/// Convergence requires |λ| ‖K‖_∞ < 1 (L^∞ operator norm).
/// The series is truncated when the increment ‖uₙ₊₁ − uₙ‖_∞ < `cfg.neumann_tol`.
///
/// Integrals are approximated with Gauss-Legendre quadrature on \[a, b\].
///
/// # Arguments
///
/// * `kernel` — kernel K(x, t)
/// * `f` — right-hand side f(x)
/// * `cfg` — solver configuration (must include `neumann_max_terms` and `neumann_tol`)
///
/// # Errors
///
/// Returns [`IntegrateError::ConvergenceError`] if the series does not converge.
pub fn fredholm_neumann(
    kernel: &dyn Fn(f64, f64) -> f64,
    f: &dyn Fn(f64) -> f64,
    cfg: &FredholmSolverConfig,
) -> IntegrateResult<FredholmSolution> {
    let a = cfg.a;
    let b = cfg.b;
    let lam = cfg.lambda;
    let n = cfg.n_quad;

    // Build quadrature grid
    let (xi_ref, wi_ref) = gauss_legendre_nodes(n);
    let half_len = (b - a) * 0.5;
    let mid = (a + b) * 0.5;
    let nodes: Vec<f64> = xi_ref.iter().map(|&xi| mid + half_len * xi).collect();
    let weights: Vec<f64> = wi_ref.iter().map(|&wi| wi * half_len).collect();
    let nq = nodes.len();

    // Evaluate f on grid
    let f_vals: Vec<f64> = nodes.iter().map(|&x| f(x)).collect();

    // Precompute kernel matrix K[i][j] = K(xᵢ, tⱼ)
    let mut k_mat = vec![vec![0.0_f64; nq]; nq];
    for i in 0..nq {
        for j in 0..nq {
            k_mat[i][j] = kernel(nodes[i], nodes[j]);
        }
    }

    // Neumann iteration: u_{n+1}[i] = f[i] + λ Σⱼ wⱼ K[i][j] u_n[j]
    let mut u_prev = f_vals.clone();

    for iter in 0..cfg.neumann_max_terms {
        let mut u_next = vec![0.0_f64; nq];
        for i in 0..nq {
            let integral: f64 = (0..nq).map(|j| weights[j] * k_mat[i][j] * u_prev[j]).sum();
            u_next[i] = f_vals[i] + lam * integral;
        }

        // Check convergence
        let max_diff = u_next
            .iter()
            .zip(u_prev.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        u_prev = u_next;

        if max_diff < cfg.neumann_tol {
            return Ok(FredholmSolution {
                nodes,
                weights,
                u: u_prev,
                condition_estimate: 0.0,
                n_terms_used: iter + 1,
            });
        }
    }

    Err(IntegrateError::ConvergenceError(format!(
        "Neumann series did not converge in {} iterations; \
         check that |λ| * ‖K‖ < 1",
        cfg.neumann_max_terms
    )))
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ------------------------------------------------------------------
    // Nyström method tests
    // ------------------------------------------------------------------

    /// Zero kernel: u = f trivially
    #[test]
    fn test_nystrom_zero_kernel() {
        let cfg = FredholmSolverConfig {
            a: 0.0,
            b: 1.0,
            n_quad: 8,
            lambda: 1.0,
            ..Default::default()
        };
        let f = |x: f64| (PI * x).sin();
        let kernel = |_x: f64, _t: f64| 0.0_f64;
        let sol = nystrom_method(&kernel, &f, &cfg).expect("nystrom zero kernel failed");
        for (&xi, &ui) in sol.nodes.iter().zip(sol.u.iter()) {
            let exact = (PI * xi).sin();
            assert!(
                (ui - exact).abs() < 1e-12,
                "u({:.3}) = {:.6} != f = {:.6}",
                xi,
                ui,
                exact
            );
        }
    }

    /// Separable kernel K(x,t) = x*t, λ=0.5
    /// Analytic: c = 1/(π*(1 - λ/3)), u(x) = sin(πx) + λ*c*x
    #[test]
    fn test_nystrom_separable_kernel() {
        let lam = 0.5_f64;
        let c_exact = 1.0 / (PI * (1.0 - lam / 3.0));
        let cfg = FredholmSolverConfig {
            a: 0.0,
            b: 1.0,
            n_quad: 8,
            lambda: lam,
            ..Default::default()
        };
        let f = |x: f64| (PI * x).sin();
        let kernel = |x: f64, t: f64| x * t;
        let sol = nystrom_method(&kernel, &f, &cfg).expect("nystrom separable failed");
        for (&xi, &ui) in sol.nodes.iter().zip(sol.u.iter()) {
            let exact = (PI * xi).sin() + lam * c_exact * xi;
            assert!(
                (ui - exact).abs() < 1e-6,
                "u({:.3}) = {:.8} != {:.8}",
                xi,
                ui,
                exact
            );
        }
    }

    /// FredholmSolver convenience wrapper
    #[test]
    fn test_solver_struct_nystrom() {
        let cfg = FredholmSolverConfig {
            a: 0.0,
            b: 1.0,
            n_quad: 8,
            lambda: 0.3,
            ..Default::default()
        };
        let solver = FredholmSolver::new(|x: f64, t: f64| (-((x - t).powi(2))).exp(), cfg);
        let f = |x: f64| 1.0_f64 + x;
        let sol = solver.nystrom_method(f).expect("solver nystrom failed");
        // Just verify it returns a valid solution (same size as n_quad)
        assert_eq!(sol.nodes.len(), 8);
        assert_eq!(sol.u.len(), 8);
    }

    // ------------------------------------------------------------------
    // Degenerate kernel tests
    // ------------------------------------------------------------------

    /// K(x,t) = 1 (rank-1 degenerate), f(x) = 1, λ = 0.5
    /// Exact: c = ∫₀¹ 1·u(t) dt;  u(t) = 1 + 0.5*c*1
    ///        c = ∫₀¹ (1 + 0.5 c) dt = 1 + 0.5 c  ⟹  c = 2, u(x) = 2
    #[test]
    fn test_degenerate_constant_kernel() {
        let cfg = FredholmSolverConfig {
            a: 0.0,
            b: 1.0,
            n_quad: 8,
            lambda: 0.5,
            ..Default::default()
        };
        let f = |_x: f64| 1.0_f64;
        let basis_a: Vec<Box<dyn Fn(f64) -> f64>> = vec![Box::new(|_x: f64| 1.0_f64)];
        let basis_b: Vec<Box<dyn Fn(f64) -> f64>> = vec![Box::new(|_t: f64| 1.0_f64)];
        let eval_pts: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
        let sol = degenerate_kernel(&cfg, &f, &basis_a, &basis_b, &eval_pts)
            .expect("degenerate constant kernel failed");
        for &ui in &sol.u {
            assert!((ui - 2.0).abs() < 1e-10, "u = {} != 2.0", ui);
        }
    }

    /// K(x,t) = x*t, λ=0.5 (matches nystrom test above)
    #[test]
    fn test_degenerate_xt_kernel() {
        let lam = 0.5_f64;
        let c_exact = 1.0 / (PI * (1.0 - lam / 3.0));
        let cfg = FredholmSolverConfig {
            a: 0.0,
            b: 1.0,
            n_quad: 10,
            lambda: lam,
            ..Default::default()
        };
        let f = |x: f64| (PI * x).sin();
        let basis_a: Vec<Box<dyn Fn(f64) -> f64>> = vec![Box::new(|x: f64| x)];
        let basis_b: Vec<Box<dyn Fn(f64) -> f64>> = vec![Box::new(|t: f64| t)];
        let eval_pts: Vec<f64> = (1..=9).map(|i| i as f64 / 10.0).collect();
        let sol = degenerate_kernel(&cfg, &f, &basis_a, &basis_b, &eval_pts)
            .expect("degenerate xt kernel failed");
        for (&xi, &ui) in sol.nodes.iter().zip(sol.u.iter()) {
            let exact = (PI * xi).sin() + lam * c_exact * xi;
            assert!(
                (ui - exact).abs() < 1e-6,
                "u({:.2}) = {:.8} != {:.8}",
                xi,
                ui,
                exact
            );
        }
    }

    // ------------------------------------------------------------------
    // Neumann series tests
    // ------------------------------------------------------------------

    /// Zero kernel: Neumann series should converge in one step
    #[test]
    fn test_neumann_zero_kernel() {
        let cfg = FredholmSolverConfig {
            a: 0.0,
            b: 1.0,
            n_quad: 8,
            lambda: 1.0,
            neumann_max_terms: 50,
            neumann_tol: 1e-12,
        };
        let f = |x: f64| x * x;
        let kernel = |_x: f64, _t: f64| 0.0_f64;
        let sol = fredholm_neumann(&kernel, &f, &cfg).expect("neumann zero kernel failed");
        for (&xi, &ui) in sol.nodes.iter().zip(sol.u.iter()) {
            assert!(
                (ui - xi * xi).abs() < 1e-12,
                "neumann zero: u({}) = {} != {}",
                xi,
                ui,
                xi * xi
            );
        }
        assert_eq!(sol.n_terms_used, 1);
    }

    /// Symmetric kernel K(x,t) = 0.1 (constant), λ=0.5, f(x)=1
    /// Neumann should agree with exact solution u(x) = 2
    #[test]
    fn test_neumann_constant_kernel() {
        let cfg = FredholmSolverConfig {
            a: 0.0,
            b: 1.0,
            n_quad: 8,
            lambda: 0.5,
            neumann_max_terms: 200,
            neumann_tol: 1e-10,
        };
        let f = |_x: f64| 1.0_f64;
        let kernel = |_x: f64, _t: f64| 1.0_f64;
        let sol = fredholm_neumann(&kernel, &f, &cfg).expect("neumann constant kernel failed");
        // u(x) = 2
        for &ui in &sol.u {
            assert!((ui - 2.0).abs() < 1e-4, "neumann constant: u = {} != 2", ui);
        }
    }

    /// Neumann interpolation consistency with Nyström
    #[test]
    fn test_neumann_vs_nystrom() {
        let lam = 0.1_f64; // small λ — fast convergence
        let cfg = FredholmSolverConfig {
            a: 0.0,
            b: 1.0,
            n_quad: 8,
            lambda: lam,
            neumann_max_terms: 100,
            neumann_tol: 1e-10,
        };
        let f = |x: f64| 1.0_f64 + 0.5 * x;
        let kernel = |x: f64, t: f64| (x * t).sin();
        let sol_nm = fredholm_neumann(&kernel, &f, &cfg).expect("neumann sin kernel");
        let sol_ny = nystrom_method(&kernel, &f, &cfg).expect("nystrom sin kernel");
        for (un, uy) in sol_nm.u.iter().zip(sol_ny.u.iter()) {
            assert!(
                (un - uy).abs() < 1e-4,
                "neumann={:.8} nystrom={:.8} differ by more than 1e-4",
                un,
                uy
            );
        }
    }
}
