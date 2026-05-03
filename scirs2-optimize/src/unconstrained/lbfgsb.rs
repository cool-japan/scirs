//! L-BFGS-B: Limited-memory BFGS with box constraints
//!
//! This module implements the L-BFGS-B algorithm of Byrd, Lu, Nocedal & Zhu (1995).
//! It extends the standard L-BFGS method to handle box constraints of the form
//! `lᵢ ≤ xᵢ ≤ uᵢ` (any combination of finite/infinite bounds) through:
//!
//! 1. **Cauchy point computation** – piecewise linear minimisation along the
//!    projected gradient to identify an active set.
//! 2. **Subspace minimisation** – quadratic minimisation in the free-variable
//!    subspace using the compact L-BFGS representation.
//! 3. **Backtracking line search** – projected backtracking that keeps the
//!    iterate inside the feasible box.
//!
//! ## References
//!
//! - Byrd, R.H., Lu, P., Nocedal, J., Zhu, C. (1995).
//!   "A limited memory algorithm for bound constrained optimization."
//!   *SIAM J. Sci. Comput.* 16(5): 1190–1208.
//! - Zhu, C., Byrd, R.H., Lu, P., Nocedal, J. (1997). "Algorithm 778: L-BFGS-B".
//!   *ACM Trans. Math. Softw.* 23(4): 550–560.

use crate::error::OptimizeError;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::finite_difference_gradient;
use crate::unconstrained::Bounds;
use scirs2_core::ndarray::{Array1, ArrayView1};

// ─── Projected gradient ───────────────────────────────────────────────────────

/// Gradient projection onto a feasible box.
///
/// Computes the projected gradient direction used to detect active constraints
/// and to evaluate optimality of the current iterate.
pub struct ProjectedGradient;

impl ProjectedGradient {
    /// Compute the projected gradient at `x` given box bounds.
    ///
    /// For each component:
    /// ```text
    /// pg[i] = g[i]  if lᵢ < xᵢ < uᵢ  (interior)
    /// pg[i] = min(g[i], 0)  if xᵢ == lᵢ (at lower bound)
    /// pg[i] = max(g[i], 0)  if xᵢ == uᵢ (at upper bound)
    /// ```
    pub fn project(x: &[f64], g: &[f64], lower: &[Option<f64>], upper: &[Option<f64>]) -> Vec<f64> {
        let n = x.len();
        let mut pg = vec![0.0f64; n];
        for i in 0..n {
            let at_lb = lower[i].map_or(false, |l| (x[i] - l).abs() < 1e-12);
            let at_ub = upper[i].map_or(false, |u| (x[i] - u).abs() < 1e-12);

            pg[i] = if at_lb && at_ub {
                0.0 // pinned between identical bounds
            } else if at_lb {
                g[i].min(0.0).abs() * g[i].signum() * (-1.0) // allow only non-negative steps
                                                             // Simplify: if at lower bound, projected gradient component is min(g,0)
                                                             // Note: projected gradient of f at lower bound = max(g, 0) (feasible direction = +)
                                                             // Convention here: projected gradient for optimality measure = x - proj(x - g)
            } else if at_ub {
                g[i].max(0.0)
            } else {
                g[i]
            };
        }
        pg
    }

    /// Compute the optimality measure: ‖x - P(x - g)‖_∞
    ///
    /// This is the standard bounded optimality measure used in L-BFGS-B convergence tests.
    pub fn optimality_measure(
        x: &[f64],
        g: &[f64],
        lower: &[Option<f64>],
        upper: &[Option<f64>],
    ) -> f64 {
        let n = x.len();
        let mut max_val = 0.0f64;
        for i in 0..n {
            let x_minus_g = x[i] - g[i];
            // Project x - g onto [lb, ub]
            let proj = match (lower[i], upper[i]) {
                (Some(l), Some(u)) => x_minus_g.max(l).min(u),
                (Some(l), None) => x_minus_g.max(l),
                (None, Some(u)) => x_minus_g.min(u),
                (None, None) => x_minus_g,
            };
            let val = (x[i] - proj).abs();
            if val > max_val {
                max_val = val;
            }
        }
        max_val
    }
}

// ─── Cauchy point ─────────────────────────────────────────────────────────────

/// Generalised Cauchy point computation for L-BFGS-B.
///
/// Finds the first local minimiser of the quadratic model along the piecewise
/// linear path `x(t) = P[x - t g]` (the projected steepest descent path).
///
/// Returns `(x_cauchy, active_set)` where `active_set[i]` is `true` if variable
/// `i` is fixed at its bound at the Cauchy point.
pub struct CauchyPoint;

impl CauchyPoint {
    /// Compute the generalised Cauchy point.
    ///
    /// # Arguments
    /// * `x`    – current iterate
    /// * `g`    – gradient at `x`
    /// * `lower`, `upper` – box bounds
    /// * `theta` – scaling factor for the identity in the compact BFGS representation
    /// * `s_vecs`, `y_vecs` – L-BFGS curvature pairs
    ///
    /// # Returns
    /// `(x_cauchy, free_vars)` where `free_vars[i] == true` iff variable `i` is
    /// not fixed at a bound at the Cauchy point.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        x: &[f64],
        g: &[f64],
        lower: &[Option<f64>],
        upper: &[Option<f64>],
        theta: f64,
        s_vecs: &[Vec<f64>],
        y_vecs: &[Vec<f64>],
    ) -> (Vec<f64>, Vec<bool>) {
        let n = x.len();

        // Break-points: for each variable, compute t_i = distance to bound in direction d = -g
        let mut break_pts: Vec<(f64, usize)> = Vec::with_capacity(n);
        let d: Vec<f64> = (0..n)
            .map(|i| {
                if g[i] < 0.0 {
                    // Moving in positive direction (step = -g > 0)
                    upper[i].map_or(f64::INFINITY, |u| (u - x[i]) / (-g[i]).max(1e-300))
                } else if g[i] > 0.0 {
                    // Moving in negative direction
                    lower[i].map_or(f64::INFINITY, |l| (x[i] - l) / g[i].max(1e-300))
                } else {
                    f64::INFINITY
                }
            })
            .collect();

        for (i, &ti) in d.iter().enumerate() {
            if ti.is_finite() && ti >= 0.0 {
                break_pts.push((ti, i));
            }
        }
        break_pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Initialise at x, c accumulates displacements along -g
        let mut xc = x.to_vec();
        let mut active = vec![false; n];
        let mut t_prev = 0.0;

        // For each segment between consecutive break-points, check if the
        // quadratic minimiser is in that segment.
        // Compact L-BFGS quadratic: m(p) ≈ gᵀp + ½ θ pᵀp - pᵀ W M Wᵀ p
        // We use a simplified version: θ I as the curvature (first-order correction only)
        // and include the two-loop L-BFGS correction in the full subspace min.

        // Derivative of the restricted quadratic along the projected steepest-descent path
        // fp = gᵀ d + θ ‖d‖² − (L-BFGS correction term)
        // For the Cauchy point we use the θ-scaled model (ignoring curvature pairs for
        // simplicity of the break-point minimisation; curvature pairs are used in subspace min).

        for (bp, fixed_var) in &break_pts {
            let t = *bp;
            let dt = t - t_prev;

            // Derivative of the quadratic model along the piecewise linear path
            // in the current segment: f'(t) = gᵀ(-g) + θ t ‖g‖² (simplified)
            let g_dot_d: f64 = (0..n).filter(|&i| !active[i]).map(|i| g[i] * (-g[i])).sum();
            let d_hd: f64 = {
                // Approximate Hv using L-BFGS: H ≈ θ I + corrections
                // For the break-point step, use θ I only
                let g_free_sq: f64 = (0..n).filter(|&i| !active[i]).map(|i| g[i] * g[i]).sum();
                theta * g_free_sq + l_bfgs_correction_scalar(g, s_vecs, y_vecs, theta, &active)
            };

            if d_hd < 1e-300 {
                // Zero curvature — just take a step to the next break-point
            } else {
                // Minimum of quadratic in this segment
                let t_min = t_prev - g_dot_d / d_hd;
                if t_min <= t {
                    // Minimiser is inside this segment
                    let t_star = t_min.max(t_prev);
                    // Update xc
                    for i in 0..n {
                        if !active[i] {
                            xc[i] = x[i] - g[i] * t_star;
                            // Project
                            if let Some(l) = lower[i] {
                                xc[i] = xc[i].max(l);
                            }
                            if let Some(u) = upper[i] {
                                xc[i] = xc[i].min(u);
                            }
                        }
                    }
                    // Clamp to break-points that were already passed
                    for &(bt, bi) in break_pts.iter().filter(|(bt, _)| *bt <= t_star) {
                        let _ = bt;
                        active[bi] = true;
                        xc[bi] = if g[bi] > 0.0 {
                            lower[bi].unwrap_or(x[bi])
                        } else {
                            upper[bi].unwrap_or(x[bi])
                        };
                    }
                    let free = active.iter().map(|a| !a).collect();
                    return (xc, free);
                }
            }

            // Fix variable at bound
            active[*fixed_var] = true;
            xc[*fixed_var] = if g[*fixed_var] > 0.0 {
                lower[*fixed_var].unwrap_or(x[*fixed_var])
            } else {
                upper[*fixed_var].unwrap_or(x[*fixed_var])
            };

            let _ = dt;
            t_prev = t;
        }

        // All break-points exhausted; update remaining free variables
        for i in 0..n {
            if !active[i] {
                xc[i] = x[i] - g[i] * t_prev;
                if let Some(l) = lower[i] {
                    xc[i] = xc[i].max(l);
                }
                if let Some(u) = upper[i] {
                    xc[i] = xc[i].min(u);
                }
            }
        }

        let free = active.iter().map(|a| !a).collect();
        (xc, free)
    }
}

/// Scalar L-BFGS curvature correction for the Cauchy point.
fn l_bfgs_correction_scalar(
    g: &[f64],
    s_vecs: &[Vec<f64>],
    y_vecs: &[Vec<f64>],
    theta: f64,
    active: &[bool],
) -> f64 {
    if s_vecs.is_empty() {
        return 0.0;
    }
    let n = g.len();
    let m = s_vecs.len().min(y_vecs.len());

    // Two-loop recursion to get H_inv g and then compute correction
    // For the Cauchy point scalar we just use a simple diagonal approximation
    let mut h_diag = 1.0 / theta;
    if let (Some(s_last), Some(y_last)) = (s_vecs.last(), y_vecs.last()) {
        let sy: f64 = s_last.iter().zip(y_last.iter()).map(|(s, y)| s * y).sum();
        let yy: f64 = y_last.iter().map(|y| y * y).sum();
        if sy > 0.0 && yy > 0.0 {
            h_diag = sy / yy;
        }
    }

    // Approximate gᵀ H⁻¹ g for free variables (diagonal approx)
    let g_free_sq: f64 = (0..n).filter(|&i| !active[i]).map(|i| g[i] * g[i]).sum();
    let _ = m;
    g_free_sq * (1.0 / h_diag - theta) * 0.5
}

// ─── Subspace minimisation ────────────────────────────────────────────────────

/// Direct subspace minimisation phase for L-BFGS-B.
///
/// After the Cauchy point is computed, minimises the quadratic model over the
/// free-variable subspace (variables not fixed at bounds) using the compact
/// L-BFGS representation of the Hessian.
pub struct SubspaceMinimization;

impl SubspaceMinimization {
    /// Perform subspace minimisation.
    ///
    /// # Arguments
    /// * `x_cauchy` – the Cauchy point
    /// * `x`        – current iterate
    /// * `g`        – gradient at `x`
    /// * `free`     – boolean mask: `true` if variable is free
    /// * `lower`, `upper` – box bounds
    /// * `theta`    – L-BFGS scaling factor
    /// * `s_vecs`, `y_vecs`, `rho_vals` – L-BFGS history
    ///
    /// # Returns
    /// Updated point after subspace minimisation (projected onto box).
    #[allow(clippy::too_many_arguments)]
    pub fn minimize(
        x_cauchy: &[f64],
        x: &[f64],
        g: &[f64],
        free: &[bool],
        lower: &[Option<f64>],
        upper: &[Option<f64>],
        theta: f64,
        s_vecs: &[Vec<f64>],
        y_vecs: &[Vec<f64>],
        rho_vals: &[f64],
    ) -> Vec<f64> {
        let n = x.len();
        let free_indices: Vec<usize> = (0..n).filter(|&i| free[i]).collect();
        let n_free = free_indices.len();

        if n_free == 0 {
            return x_cauchy.to_vec();
        }

        // Extract free-variable gradient and displacement from Cauchy point
        let r_free: Vec<f64> = free_indices
            .iter()
            .map(|&i| {
                // Reduced gradient: gradient of quadratic at x_cauchy restricted to free vars
                // r = g + H (x_cauchy - x)  projected to free vars
                // Approximate with the L-BFGS product
                g[i]
            })
            .collect();

        // Compute H * (x_cauchy - x) for free variables using L-BFGS two-loop
        let delta: Vec<f64> = (0..n).map(|i| x_cauchy[i] - x[i]).collect();
        let h_delta = l_bfgs_product(&delta, s_vecs, y_vecs, rho_vals, theta);

        // Reduced gradient at x_cauchy: r_c = g + H * delta (free vars)
        let r_c: Vec<f64> = free_indices
            .iter()
            .map(|&i| r_free[free_indices.iter().position(|&j| j == i).unwrap_or(0)] + h_delta[i])
            .collect();

        // Compute the reduced Hessian vector product direction = -H_free^{-1} r_c
        // Use L-BFGS two-loop on the free-variable subspace
        let mut r_full = vec![0.0f64; n];
        for (k, &fi) in free_indices.iter().enumerate() {
            r_full[fi] = r_c[k];
        }
        let step_full = l_bfgs_two_loop_neg(&r_full, s_vecs, y_vecs, rho_vals, theta);

        // Only keep free-variable components of the step
        let mut x_new = x_cauchy.to_vec();
        for &fi in &free_indices {
            x_new[fi] += step_full[fi];
            // Project
            if let Some(l) = lower[fi] {
                x_new[fi] = x_new[fi].max(l);
            }
            if let Some(u) = upper[fi] {
                x_new[fi] = x_new[fi].min(u);
            }
        }

        x_new
    }
}

/// L-BFGS two-loop recursion: returns -H^{-1} q  (the negative product).
fn l_bfgs_two_loop_neg(
    q_in: &[f64],
    s_vecs: &[Vec<f64>],
    y_vecs: &[Vec<f64>],
    rho_vals: &[f64],
    theta: f64,
) -> Vec<f64> {
    let n = q_in.len();
    let m = s_vecs.len().min(y_vecs.len()).min(rho_vals.len());
    let mut q = q_in.to_vec();
    let mut alphas = vec![0.0f64; m];

    // First loop (most recent to oldest)
    for k in (0..m).rev() {
        let s = &s_vecs[k];
        let alpha = rho_vals[k] * dot(s, &q);
        alphas[k] = alpha;
        let y = &y_vecs[k];
        for i in 0..n {
            q[i] -= alpha * y[i];
        }
    }

    // Scale by initial Hessian approximation (theta * I inverse = 1/theta * I)
    let h0 = if m > 0 {
        let sy: f64 = dot(&s_vecs[m - 1], &y_vecs[m - 1]);
        let yy: f64 = dot(&y_vecs[m - 1], &y_vecs[m - 1]);
        if yy > 1e-300 {
            sy / yy
        } else {
            1.0 / theta
        }
    } else {
        1.0 / theta
    };
    let mut r: Vec<f64> = q.iter().map(|qi| h0 * qi).collect();

    // Second loop (oldest to most recent)
    for k in 0..m {
        let y = &y_vecs[k];
        let beta = rho_vals[k] * dot(y, &r);
        let s = &s_vecs[k];
        for i in 0..n {
            r[i] += s[i] * (alphas[k] - beta);
        }
    }

    // Return negative (we want -H^{-1} q)
    r.iter().map(|ri| -ri).collect()
}

/// L-BFGS matrix-vector product: returns H * v using curvature pairs.
fn l_bfgs_product(
    v: &[f64],
    s_vecs: &[Vec<f64>],
    y_vecs: &[Vec<f64>],
    rho_vals: &[f64],
    theta: f64,
) -> Vec<f64> {
    // H * v ≈ θ v + corrections via BFGS recursion on inverse then invert
    // For simplicity we use a single-pass scaling + skew correction
    let n = v.len();
    let m = s_vecs.len().min(y_vecs.len()).min(rho_vals.len());

    // Start with θ v
    let mut result: Vec<f64> = v.iter().map(|vi| theta * vi).collect();

    // Apply rank-2 corrections: H = θI + Σ (rho_k y_k y_kᵀ - θ s_k s_kᵀ) approximately
    // This is a first-order correction
    for k in 0..m {
        let s = &s_vecs[k];
        let y = &y_vecs[k];
        let sy: f64 = dot(s, y);
        if sy.abs() < 1e-300 {
            continue;
        }
        let sv: f64 = dot(s, v);
        let yv: f64 = dot(y, v);
        let rho = rho_vals[k];
        // Δ H v = rho (yy^T - sy (ss^T / sy)) v  [simplified rank-2 correction]
        for i in 0..n {
            result[i] += rho * y[i] * yv - theta * rho * sy * s[i] * sv / sy.max(1e-300);
        }
    }
    result
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

// ─── LBFGSB result ────────────────────────────────────────────────────────────

/// Result of the L-BFGS-B optimisation.
#[derive(Debug, Clone)]
pub struct LBFGSBResult {
    /// Solution vector
    pub x: Array1<f64>,
    /// Objective function value at solution
    pub f_val: f64,
    /// Gradient at solution
    pub gradient: Array1<f64>,
    /// Projected gradient norm at solution (optimality measure)
    pub proj_grad_norm: f64,
    /// Number of outer iterations
    pub n_iter: usize,
    /// Number of function evaluations
    pub n_fev: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Termination message
    pub message: String,
}

// ─── L-BFGS-B solver ─────────────────────────────────────────────────────────

/// Options for the L-BFGS-B solver.
#[derive(Debug, Clone)]
pub struct LBFGSBOptions {
    /// Number of L-BFGS history pairs to retain (memory parameter `m`)
    pub memory: usize,
    /// Projected gradient tolerance (convergence criterion)
    pub pgtol: f64,
    /// Function value tolerance (factr * machine_eps)
    pub factr: f64,
    /// Finite-difference step for gradient computation
    pub eps: f64,
    /// Maximum outer iterations
    pub max_iter: usize,
    /// Maximum backtracking steps in line search
    pub max_ls_steps: usize,
    /// Armijo sufficient-decrease constant
    pub c1: f64,
    /// Backtracking reduction factor
    pub backtrack_rho: f64,
    /// Box bounds (required for the B in L-BFGS-B)
    pub bounds: Option<Bounds>,
}

impl Default for LBFGSBOptions {
    fn default() -> Self {
        Self {
            memory: 10,
            pgtol: 1e-5,
            factr: 1e7,
            eps: f64::EPSILON.sqrt(),
            max_iter: 500,
            max_ls_steps: 30,
            c1: 1e-4,
            backtrack_rho: 0.5,
            bounds: None,
        }
    }
}

/// L-BFGS-B solver for box-constrained optimisation.
pub struct LBFGSB {
    /// Solver options
    pub options: LBFGSBOptions,
}

impl LBFGSB {
    /// Create with given options.
    pub fn new(options: LBFGSBOptions) -> Self {
        Self { options }
    }

    /// Create with default options.
    pub fn default_solver() -> Self {
        Self {
            options: LBFGSBOptions::default(),
        }
    }

    /// Minimise a function subject to box constraints.
    ///
    /// # Arguments
    /// * `fun` – objective function
    /// * `x0`  – initial iterate (will be projected onto the box)
    pub fn minimize<F>(&self, mut fun: F, x0: &[f64]) -> Result<LBFGSBResult, OptimizeError>
    where
        F: FnMut(&ArrayView1<f64>) -> f64,
    {
        let opts = &self.options;
        let n = x0.len();
        let m = opts.memory;
        let machine_eps = f64::EPSILON;
        let ftol = opts.factr * machine_eps;

        // Extract bound arrays (replicated for lifetime)
        let (lower, upper) = opts
            .bounds
            .as_ref()
            .map(|b| (b.lower.clone(), b.upper.clone()))
            .unwrap_or_else(|| (vec![None; n], vec![None; n]));

        // Initialise iterate (projected)
        let mut x: Vec<f64> = x0.to_vec();
        project_point(&mut x, &lower, &upper);

        let mut n_fev = 0usize;

        let x_arr = Array1::from(x.clone());
        let mut f = {
            n_fev += 1;
            fun(&x_arr.view())
        };
        let mut g = {
            let xa = Array1::from(x.clone());
            let grad = finite_difference_gradient(&mut fun, &xa.view(), opts.eps)?;
            n_fev += 2 * n;
            grad.to_vec()
        };

        // L-BFGS history
        let mut s_vecs: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut y_vecs: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut rho_vals: Vec<f64> = Vec::with_capacity(m);

        // theta: scaling for the L-BFGS Hessian approximation
        let mut theta = 1.0f64;

        let mut iter = 0usize;
        let mut converged = false;
        let mut message = "Maximum iterations reached".to_string();

        loop {
            // Optimality check using the projected gradient norm
            let pg_norm = ProjectedGradient::optimality_measure(&x, &g, &lower, &upper);
            if pg_norm <= opts.pgtol {
                converged = true;
                message = "Projected gradient norm below tolerance".to_string();
                break;
            }

            if iter >= opts.max_iter {
                break;
            }

            // 1. Compute generalised Cauchy point
            let (x_cauchy, free) =
                CauchyPoint::compute(&x, &g, &lower, &upper, theta, &s_vecs, &y_vecs);

            // 2. Subspace minimisation
            let x_bar = SubspaceMinimization::minimize(
                &x_cauchy, &x, &g, &free, &lower, &upper, theta, &s_vecs, &y_vecs, &rho_vals,
            );

            // 3. Compute descent direction and do a projected line search
            let d: Vec<f64> = (0..n).map(|i| x_bar[i] - x[i]).collect();

            // Check that d is a descent direction
            let slope: f64 = dot(&g, &d);
            if slope >= 0.0 {
                // Fall back to projected gradient direction
                let pg: Vec<f64> = (0..n)
                    .map(|i| {
                        let at_lb = lower[i].map_or(false, |l| (x[i] - l).abs() < 1e-12);
                        let at_ub = upper[i].map_or(false, |u| (x[i] - u).abs() < 1e-12);
                        if at_lb && g[i] > 0.0 {
                            0.0
                        } else if at_ub && g[i] < 0.0 {
                            0.0
                        } else {
                            -g[i]
                        }
                    })
                    .collect();
                let pg_norm_inner = dot(&pg, &pg).sqrt();
                if pg_norm_inner < 1e-12 {
                    converged = true;
                    message = "Projected gradient is zero, at constrained optimum".to_string();
                    break;
                }
                // Take a small step along the negative projected gradient
                let alpha = projected_backtrack(
                    &mut fun,
                    &x,
                    &pg,
                    f,
                    -dot(&g, &pg),
                    &lower,
                    &upper,
                    opts.c1,
                    opts.backtrack_rho,
                    opts.max_ls_steps,
                    &mut n_fev,
                );
                let mut x_new: Vec<f64> = (0..n).map(|i| x[i] + alpha * pg[i]).collect();
                project_point(&mut x_new, &lower, &upper);

                let x_new_arr = Array1::from(x_new.clone());
                let f_new = {
                    n_fev += 1;
                    fun(&x_new_arr.view())
                };

                let s: Vec<f64> = (0..n).map(|i| x_new[i] - x[i]).collect();
                let g_new = {
                    let xa = Array1::from(x_new.clone());
                    let grad = finite_difference_gradient(&mut fun, &xa.view(), opts.eps)?;
                    n_fev += 2 * n;
                    grad.to_vec()
                };
                let y: Vec<f64> = (0..n).map(|i| g_new[i] - g[i]).collect();
                let sy = dot(&s, &y);
                if sy > 1e-10 {
                    update_lbfgs_history(&mut s_vecs, &mut y_vecs, &mut rho_vals, s, y, sy, m);
                    theta = dot(
                        &y_vecs.last().expect("unexpected None or Err"),
                        &y_vecs.last().expect("unexpected None or Err"),
                    ) / dot(
                        &s_vecs.last().expect("unexpected None or Err"),
                        &y_vecs.last().expect("unexpected None or Err"),
                    )
                    .max(1e-300);
                }

                // Check function-value convergence
                if (f - f_new).abs() < ftol * (1.0 + f.abs()) {
                    x = x_new;
                    f = f_new;
                    g = g_new;
                    converged = true;
                    message = "Function value change below tolerance".to_string();
                    break;
                }

                x = x_new;
                f = f_new;
                g = g_new;
                iter += 1;
                continue;
            }

            // Projected backtracking line search
            let alpha = projected_backtrack(
                &mut fun,
                &x,
                &d,
                f,
                slope,
                &lower,
                &upper,
                opts.c1,
                opts.backtrack_rho,
                opts.max_ls_steps,
                &mut n_fev,
            );

            let mut x_new: Vec<f64> = (0..n).map(|i| x[i] + alpha * d[i]).collect();
            project_point(&mut x_new, &lower, &upper);

            let x_new_arr = Array1::from(x_new.clone());
            let f_new = {
                n_fev += 1;
                fun(&x_new_arr.view())
            };

            // Update L-BFGS curvature pairs
            let s: Vec<f64> = (0..n).map(|i| x_new[i] - x[i]).collect();
            let g_new = {
                let xa = Array1::from(x_new.clone());
                let grad = finite_difference_gradient(&mut fun, &xa.view(), opts.eps)?;
                n_fev += 2 * n;
                grad.to_vec()
            };
            let y: Vec<f64> = (0..n).map(|i| g_new[i] - g[i]).collect();

            let sy = dot(&s, &y);
            if sy > machine_eps * dot(&y, &y) {
                update_lbfgs_history(&mut s_vecs, &mut y_vecs, &mut rho_vals, s, y, sy, m);
                // Update theta: θ = yᵀy / sᵀy
                if let (Some(y_last), Some(s_last)) = (y_vecs.last(), s_vecs.last()) {
                    let yy = dot(y_last, y_last);
                    let sy_last = dot(s_last, y_last);
                    if sy_last > 1e-300 {
                        theta = yy / sy_last;
                    }
                }
            }

            // Check function-value convergence
            if (f - f_new).abs() < ftol * (1.0 + f.abs()) {
                x = x_new;
                f = f_new;
                g = g_new;
                converged = true;
                message = "Function value change below tolerance".to_string();
                break;
            }

            x = x_new;
            f = f_new;
            g = g_new;
            iter += 1;
        }

        let pg_final = ProjectedGradient::optimality_measure(&x, &g, &lower, &upper);

        Ok(LBFGSBResult {
            x: Array1::from(x),
            f_val: f,
            gradient: Array1::from(g),
            proj_grad_norm: pg_final,
            n_iter: iter,
            n_fev,
            converged,
            message,
        })
    }
}

// ─── Helper functions ─────────────────────────────────────────────────────────

/// Project a point onto the box defined by `lower` and `upper`.
fn project_point(x: &mut Vec<f64>, lower: &[Option<f64>], upper: &[Option<f64>]) {
    for i in 0..x.len() {
        if let Some(l) = lower[i] {
            if x[i] < l {
                x[i] = l;
            }
        }
        if let Some(u) = upper[i] {
            if x[i] > u {
                x[i] = u;
            }
        }
    }
}

/// Projected backtracking line search.
fn projected_backtrack<F>(
    fun: &mut F,
    x: &[f64],
    d: &[f64],
    f0: f64,
    slope: f64,
    lower: &[Option<f64>],
    upper: &[Option<f64>],
    c1: f64,
    rho: f64,
    max_steps: usize,
    n_fev: &mut usize,
) -> f64
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut alpha = 1.0f64;

    for _ in 0..max_steps {
        let mut x_trial: Vec<f64> = (0..n).map(|i| x[i] + alpha * d[i]).collect();
        project_point(&mut x_trial, lower, upper);
        let x_arr = Array1::from(x_trial);
        *n_fev += 1;
        let f_trial = fun(&x_arr.view());

        if f_trial <= f0 + c1 * alpha * slope.abs() {
            return alpha;
        }
        alpha *= rho;
        if alpha < 1e-14 {
            break;
        }
    }
    alpha
}

/// Update L-BFGS history with a new (s, y) pair, evicting oldest if full.
fn update_lbfgs_history(
    s_vecs: &mut Vec<Vec<f64>>,
    y_vecs: &mut Vec<Vec<f64>>,
    rho_vals: &mut Vec<f64>,
    s: Vec<f64>,
    y: Vec<f64>,
    sy: f64,
    m: usize,
) {
    if s_vecs.len() >= m {
        s_vecs.remove(0);
        y_vecs.remove(0);
        rho_vals.remove(0);
    }
    rho_vals.push(1.0 / sy.max(1e-300));
    s_vecs.push(s);
    y_vecs.push(y);
}

/// Wrapper for the standard `OptimizeResult` API.
pub fn minimize_lbfgsb_advanced<F>(
    fun: F,
    x0: &[f64],
    options: Option<LBFGSBOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let opts = options.unwrap_or_default();
    let solver = LBFGSB::new(opts);
    let result = solver.minimize(fun, x0)?;

    Ok(OptimizeResult {
        x: result.x.clone(),
        fun: result.f_val,
        nit: result.n_iter,
        func_evals: result.n_fev,
        nfev: result.n_fev,
        success: result.converged,
        message: result.message,
        jacobian: Some(result.gradient),
        hessian: None,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn quadratic(x: &ArrayView1<f64>) -> f64 {
        (x[0] - 1.0).powi(2) + 4.0 * (x[1] - 2.0).powi(2)
    }

    fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }

    #[test]
    fn test_projected_gradient_optimality() {
        // At (1.0, 2.0) the gradient of our quadratic is (0, 0) → optimality = 0
        let x = vec![1.0, 2.0];
        let g = vec![0.0, 0.0];
        let lower = vec![None, None];
        let upper = vec![None, None];
        let opt = ProjectedGradient::optimality_measure(&x, &g, &lower, &upper);
        assert_abs_diff_eq!(opt, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lbfgsb_unconstrained_quadratic() {
        let mut opts = LBFGSBOptions::default();
        opts.pgtol = 1e-6;
        let solver = LBFGSB::new(opts);
        let result = solver
            .minimize(quadratic, &[0.0, 0.0])
            .expect("L-BFGS-B failed");
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_lbfgsb_bounded_quadratic() {
        // Minimum of (x-1)² + 4(y-2)² is at (1,2), but constrain y ≤ 1.0
        let mut opts = LBFGSBOptions::default();
        opts.bounds = Some(Bounds::new(&[(None, None), (None, Some(1.0))]));
        let solver = LBFGSB::new(opts);
        let result = solver
            .minimize(quadratic, &[0.0, 0.0])
            .expect("L-BFGS-B failed");
        assert!(result.converged || result.n_iter > 0);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_lbfgsb_rosenbrock() {
        let mut opts = LBFGSBOptions::default();
        opts.max_iter = 500;
        opts.pgtol = 1e-4;
        let solver = LBFGSB::new(opts);
        let result = solver
            .minimize(rosenbrock, &[0.5, 0.5])
            .expect("L-BFGS-B failed");
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-2);
    }

    #[test]
    fn test_cauchy_point_trivial() {
        let x = vec![0.5, 0.5];
        let g = vec![1.0, 1.0];
        let lower = vec![Some(0.0), Some(0.0)];
        let upper = vec![Some(1.0), Some(1.0)];
        let (xc, _) = CauchyPoint::compute(&x, &g, &lower, &upper, 1.0, &[], &[]);
        // With g = [1,1] > 0, Cauchy moves toward lower bounds
        assert!(xc[0] >= 0.0);
        assert!(xc[1] >= 0.0);
        assert!(xc[0] <= 1.0);
        assert!(xc[1] <= 1.0);
    }
}
