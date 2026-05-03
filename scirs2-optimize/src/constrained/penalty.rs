//! Penalty Methods for Constrained Optimization
//!
//! Penalty methods transform constrained problems into a sequence of unconstrained
//! problems by adding a penalty term to the objective function that penalizes
//! constraint violations.
//!
//! Two major classes are implemented:
//!
//! ## External Penalty Method
//! Adds a penalty for constraint violations:
//! ```text
//! P(x, mu) = f(x) + mu * sum_i max(0, g_i(x))^2 + mu * sum_j h_j(x)^2
//! ```
//! The penalty parameter mu is increased until the constraints are satisfied.
//!
//! ## Interior Penalty (Barrier) Method
//! Adds a barrier function that prevents leaving the feasible region:
//! ```text
//! B(x, mu) = f(x) - mu * sum_i ln(-g_i(x))   (log barrier)
//! ```
//! The barrier parameter mu is decreased to zero.
//!
//! # References
//! - Fiacco, A.V. & McCormick, G.P. (1968). "Nonlinear Programming: Sequential
//!   Unconstrained Minimization Techniques." SIAM.
//! - Nocedal, J. & Wright, S.J. (2006). "Numerical Optimization." Chapter 17.

use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use scirs2_core::ndarray::{Array1, ArrayView1};

/// Type of penalty method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PenaltyKind {
    /// External penalty: quadratic penalty for violations
    External,
    /// Interior penalty (log-barrier): only works from strictly feasible start
    Interior,
    /// Exact L1 penalty: |max(0, g_i(x))| + |h_j(x)|
    ExactL1,
}

/// Options for the penalty method
#[derive(Debug, Clone)]
pub struct PenaltyOptions {
    /// Penalty method kind
    pub kind: PenaltyKind,
    /// Initial penalty parameter
    pub mu_init: f64,
    /// Maximum penalty parameter (external) / min barrier (interior)
    pub mu_max: f64,
    /// Penalty increase factor (external) or decrease factor (interior)
    pub mu_factor: f64,
    /// Maximum number of outer iterations
    pub max_outer_iter: usize,
    /// Tolerance for constraint violation
    pub constraint_tol: f64,
    /// Tolerance for optimality of subproblem
    pub optimality_tol: f64,
    /// Finite difference step for gradient
    pub eps: f64,
}

impl Default for PenaltyOptions {
    fn default() -> Self {
        PenaltyOptions {
            kind: PenaltyKind::External,
            mu_init: 1.0,
            mu_max: 1e10,
            mu_factor: 10.0,
            max_outer_iter: 100,
            constraint_tol: 1e-6,
            optimality_tol: 1e-8,
            eps: 1e-7,
        }
    }
}

/// Result from penalty method
#[derive(Debug, Clone)]
pub struct PenaltyResult {
    /// Optimal solution
    pub x: Array1<f64>,
    /// Optimal objective value
    pub fun: f64,
    /// Number of outer iterations
    pub nit: usize,
    /// Total number of function evaluations
    pub nfev: usize,
    /// Success flag
    pub success: bool,
    /// Status message
    pub message: String,
    /// Final penalty parameter
    pub mu: f64,
    /// Final constraint violation
    pub constraint_violation: f64,
}

impl From<PenaltyResult> for OptimizeResults<f64> {
    fn from(r: PenaltyResult) -> Self {
        OptimizeResults {
            x: r.x,
            fun: r.fun,
            jac: None,
            hess: None,
            constr: None,
            nit: r.nit,
            nfev: r.nfev,
            njev: 0,
            nhev: 0,
            maxcv: 0,
            message: r.message,
            success: r.success,
            status: if r.success { 0 } else { 1 },
        }
    }
}

/// Internal gradient-based minimizer for penalty subproblems.
/// Uses L-BFGS-like updates with simple backtracking line search.
fn minimize_penalty_subproblem<P>(
    penalty_fn: P,
    x0: &[f64],
    max_iter: usize,
    gtol: f64,
    eps: f64,
    nfev: &mut usize,
) -> Vec<f64>
where
    P: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let m = 5usize; // L-BFGS history

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    let compute_grad = |xv: &[f64], nfev: &mut usize| -> Vec<f64> {
        let h = eps;
        let mut g = vec![0.0; n];
        let mut xp = xv.to_vec();
        let mut xm = xv.to_vec();
        *nfev += 2 * n;
        for i in 0..n {
            xp[i] = xv[i] + h;
            xm[i] = xv[i] - h;
            g[i] = (penalty_fn(&xp) - penalty_fn(&xm)) / (2.0 * h);
            xp[i] = xv[i];
            xm[i] = xv[i];
        }
        g
    };

    let mut g = compute_grad(&x, nfev);

    for _iter in 0..max_iter {
        let gnorm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < gtol {
            break;
        }

        // L-BFGS two-loop recursion
        let mut q = g.clone();
        let hist_len = s_hist.len();
        let mut alpha_hist = vec![0.0_f64; hist_len];

        for i in (0..hist_len).rev() {
            let si = &s_hist[i];
            let yi = &y_hist[i];
            let rho_i = rho_hist[i];
            let dot: f64 = si.iter().zip(q.iter()).map(|(&s, &qi)| s * qi).sum();
            alpha_hist[i] = rho_i * dot;
            let a = alpha_hist[i];
            for j in 0..n {
                q[j] -= a * yi[j];
            }
        }

        // Initial Hessian scaling
        let mut r = if hist_len > 0 {
            let last_s = &s_hist[hist_len - 1];
            let last_y = &y_hist[hist_len - 1];
            let sy: f64 = last_s.iter().zip(last_y.iter()).map(|(&s, &y)| s * y).sum();
            let yy: f64 = last_y.iter().map(|v| v * v).sum();
            let scale = if yy > 1e-15 { sy / yy } else { 1.0 };
            q.iter().map(|&qi| scale * qi).collect::<Vec<f64>>()
        } else {
            q.clone()
        };

        for i in 0..hist_len {
            let si = &s_hist[i];
            let yi = &y_hist[i];
            let rho_i = rho_hist[i];
            let dot: f64 = yi.iter().zip(r.iter()).map(|(&y, &ri)| y * ri).sum();
            let beta = rho_i * dot;
            let diff = alpha_hist[i] - beta;
            for j in 0..n {
                r[j] += si[j] * diff;
            }
        }

        // Direction: d = -r
        let d: Vec<f64> = r.iter().map(|v| -v).collect();

        // Backtracking line search
        *nfev += 1;
        let fx = penalty_fn(&x);
        let dg: f64 = d.iter().zip(g.iter()).map(|(&di, &gi)| di * gi).sum();
        let mut alpha = 1.0_f64;

        for _ls in 0..20 {
            let xnew: Vec<f64> = x
                .iter()
                .zip(d.iter())
                .map(|(&xi, &di)| xi + alpha * di)
                .collect();
            *nfev += 1;
            let fnew = penalty_fn(&xnew);
            if fnew <= fx + 1e-4 * alpha * dg.min(0.0) {
                break;
            }
            alpha *= 0.5;
        }

        let xnew: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(&xi, &di)| xi + alpha * di)
            .collect();
        let gnew = compute_grad(&xnew, nfev);

        // L-BFGS update
        let s: Vec<f64> = xnew
            .iter()
            .zip(x.iter())
            .map(|(&xni, &xi)| xni - xi)
            .collect();
        let y: Vec<f64> = gnew
            .iter()
            .zip(g.iter())
            .map(|(&gni, &gi)| gni - gi)
            .collect();
        let sy: f64 = s.iter().zip(y.iter()).map(|(&si, &yi)| si * yi).sum();

        if sy > 1e-10 {
            if s_hist.len() >= m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / sy);
        }

        x = xnew;
        g = gnew;
    }

    x
}

/// Penalty method solver
pub struct PenaltyMethod {
    pub options: PenaltyOptions,
}

impl PenaltyMethod {
    /// Create with default options
    pub fn new() -> Self {
        PenaltyMethod {
            options: PenaltyOptions::default(),
        }
    }

    /// Create with custom options
    pub fn with_options(options: PenaltyOptions) -> Self {
        PenaltyMethod { options }
    }

    /// Compute constraint violation at x
    fn compute_violation<E, G>(&self, x: &[f64], eq_cons: &[E], ineq_cons: &[G]) -> f64
    where
        E: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        let eq_viol: f64 = eq_cons.iter().map(|e| e(x).powi(2)).sum();
        let ineq_viol: f64 = ineq_cons.iter().map(|g| g(x).max(0.0).powi(2)).sum();
        (eq_viol + ineq_viol).sqrt()
    }

    /// Solve with equality and inequality constraints.
    ///
    /// # Arguments
    /// - `f`: Objective function taking a `&[f64]`
    /// - `eq_cons`: Equality constraint functions h_j(x) = 0 (slices)
    /// - `ineq_cons`: Inequality constraint functions g_i(x) <= 0 (slices)
    /// - `x0`: Initial point
    pub fn solve_slice<F, E, G>(
        &self,
        f: F,
        eq_cons: &[E],
        ineq_cons: &[G],
        x0: &[f64],
    ) -> OptimizeResult<PenaltyResult>
    where
        F: Fn(&[f64]) -> f64,
        E: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        match self.options.kind {
            PenaltyKind::External => self.solve_external_slice(f, eq_cons, ineq_cons, x0),
            PenaltyKind::Interior => self.solve_interior_slice(f, ineq_cons, x0),
            PenaltyKind::ExactL1 => self.solve_l1_slice(f, eq_cons, ineq_cons, x0),
        }
    }

    /// Solve with ArrayView1 interface (for compatibility)
    pub fn solve<F, E, G>(
        &self,
        f: F,
        eq_cons: &[E],
        ineq_cons: &[G],
        x0: &Array1<f64>,
    ) -> OptimizeResult<PenaltyResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        E: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> f64,
    {
        // Wrap ArrayView1 closures to slice-based
        let f_slice = |x: &[f64]| {
            let arr = Array1::from_vec(x.to_vec());
            f(&arr.view())
        };
        let eq_slice: Vec<Box<dyn Fn(&[f64]) -> f64>> = eq_cons
            .iter()
            .map(|e| {
                Box::new(move |x: &[f64]| {
                    let arr = Array1::from_vec(x.to_vec());
                    e(&arr.view())
                }) as Box<dyn Fn(&[f64]) -> f64>
            })
            .collect();
        let ineq_slice: Vec<Box<dyn Fn(&[f64]) -> f64>> = ineq_cons
            .iter()
            .map(|g| {
                Box::new(move |x: &[f64]| {
                    let arr = Array1::from_vec(x.to_vec());
                    g(&arr.view())
                }) as Box<dyn Fn(&[f64]) -> f64>
            })
            .collect();

        let x0_slice: Vec<f64> = x0.iter().copied().collect();
        self.solve_slice(f_slice, &eq_slice, &ineq_slice, &x0_slice)
    }

    fn solve_external_slice<F, E, G>(
        &self,
        f: F,
        eq_cons: &[E],
        ineq_cons: &[G],
        x0: &[f64],
    ) -> OptimizeResult<PenaltyResult>
    where
        F: Fn(&[f64]) -> f64,
        E: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        let mut x = x0.to_vec();
        let mut mu = self.options.mu_init;
        let mut nfev_total = 0usize;
        let mut nit = 0usize;

        for _outer in 0..self.options.max_outer_iter {
            nit += 1;
            let mu_local = mu;

            // Build penalty function capturing current state
            let penalty_at = |xv: &[f64]| -> f64 {
                let obj = f(xv);
                let penalty: f64 = eq_cons
                    .iter()
                    .map(|e| mu_local * e(xv).powi(2))
                    .sum::<f64>()
                    + ineq_cons
                        .iter()
                        .map(|g| mu_local * g(xv).max(0.0).powi(2))
                        .sum::<f64>();
                obj + penalty
            };

            let new_x = minimize_penalty_subproblem(
                penalty_at,
                &x,
                1000,
                self.options.optimality_tol,
                self.options.eps,
                &mut nfev_total,
            );
            x = new_x;

            // Check convergence after minimization
            let cv = self.compute_violation(&x, eq_cons, ineq_cons);
            if cv <= self.options.constraint_tol {
                let fun = f(&x);
                return Ok(PenaltyResult {
                    x: Array1::from_vec(x),
                    fun,
                    nit,
                    nfev: nfev_total,
                    success: true,
                    message: "Converged: constraint violation below tolerance".to_string(),
                    mu,
                    constraint_violation: cv,
                });
            }

            mu = (mu * self.options.mu_factor).min(self.options.mu_max);
        }

        let cv = self.compute_violation(&x, eq_cons, ineq_cons);
        let fun = f(&x);
        let success = cv <= self.options.constraint_tol;
        Ok(PenaltyResult {
            x: Array1::from_vec(x),
            fun,
            nit,
            nfev: nfev_total,
            success,
            message: if success {
                "Converged".to_string()
            } else {
                format!("Maximum outer iterations reached (cv={:.2e})", cv)
            },
            mu,
            constraint_violation: cv,
        })
    }

    fn solve_interior_slice<F, G>(
        &self,
        f: F,
        ineq_cons: &[G],
        x0: &[f64],
    ) -> OptimizeResult<PenaltyResult>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        // Verify strict feasibility at starting point
        for g in ineq_cons.iter() {
            if g(x0) >= 0.0 {
                return Err(OptimizeError::InvalidInput(
                    "Interior penalty requires strictly feasible starting point (g_i(x0) < 0)"
                        .to_string(),
                ));
            }
        }

        let mut x = x0.to_vec();
        let mut mu = self.options.mu_init;
        let mut nfev_total = 0usize;
        let mut nit = 0usize;

        for _outer in 0..self.options.max_outer_iter {
            nit += 1;
            if mu < 1e-12 {
                break;
            }

            let mu_local = mu;

            let barrier_fn = |xv: &[f64]| -> f64 {
                let obj = f(xv);
                let barrier: f64 = ineq_cons
                    .iter()
                    .map(|g| {
                        let gv = g(xv);
                        if gv < -1e-15 {
                            -mu_local * gv.abs().ln()
                        } else {
                            f64::INFINITY
                        }
                    })
                    .sum();
                obj + barrier
            };

            let new_x = minimize_penalty_subproblem(
                barrier_fn,
                &x,
                500,
                mu * 0.01,
                self.options.eps,
                &mut nfev_total,
            );

            // Accept only if still feasible
            let feasible = ineq_cons.iter().all(|g| g(&new_x) < 0.0);
            if feasible {
                x = new_x;
            }

            mu /= self.options.mu_factor;
        }

        let fun = f(&x);
        Ok(PenaltyResult {
            fun,
            x: Array1::from_vec(x),
            nit,
            nfev: nfev_total,
            success: true,
            message: "Interior penalty completed".to_string(),
            mu,
            constraint_violation: 0.0,
        })
    }

    fn solve_l1_slice<F, E, G>(
        &self,
        f: F,
        eq_cons: &[E],
        ineq_cons: &[G],
        x0: &[f64],
    ) -> OptimizeResult<PenaltyResult>
    where
        F: Fn(&[f64]) -> f64,
        E: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        let mut x = x0.to_vec();
        let mu = self.options.mu_init;
        let mut nfev_total = 0usize;
        let mut nit = 0usize;

        for _outer in 0..self.options.max_outer_iter {
            nit += 1;

            let l1_fn = |xv: &[f64]| -> f64 {
                let obj = f(xv);
                let penalty: f64 = eq_cons.iter().map(|e| mu * e(xv).abs()).sum::<f64>()
                    + ineq_cons.iter().map(|g| mu * g(xv).max(0.0)).sum::<f64>();
                obj + penalty
            };

            let new_x = minimize_penalty_subproblem(
                l1_fn,
                &x,
                1000,
                self.options.optimality_tol,
                self.options.eps,
                &mut nfev_total,
            );
            x = new_x;

            // Check convergence
            let eq_cv: f64 = eq_cons.iter().map(|e| e(&x).abs()).sum();
            let ineq_cv: f64 = ineq_cons.iter().map(|g| g(&x).max(0.0)).sum();
            let cv = eq_cv + ineq_cv;

            if cv <= self.options.constraint_tol {
                let fun = f(&x);
                return Ok(PenaltyResult {
                    x: Array1::from_vec(x),
                    fun,
                    nit,
                    nfev: nfev_total,
                    success: true,
                    message: "L1 penalty converged".to_string(),
                    mu,
                    constraint_violation: cv,
                });
            }
        }

        let eq_cv: f64 = eq_cons.iter().map(|e| e(&x).abs()).sum();
        let ineq_cv: f64 = ineq_cons.iter().map(|g| g(&x).max(0.0)).sum();
        let cv = eq_cv + ineq_cv;
        let fun = f(&x);
        Ok(PenaltyResult {
            x: Array1::from_vec(x),
            fun,
            nit,
            nfev: nfev_total,
            success: cv <= self.options.constraint_tol,
            message: "L1 penalty max iterations reached".to_string(),
            mu,
            constraint_violation: cv,
        })
    }
}

impl Default for PenaltyMethod {
    fn default() -> Self {
        PenaltyMethod::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PenaltyMethodKind enum (Static / Dynamic / Adaptive / AugmentedLagrangian)
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for managing the penalty parameter sequence.
///
/// This enum governs *how* the penalty coefficient is updated across outer
/// iterations, distinct from [`PenaltyKind`] which controls the *form* of the
/// penalty term (quadratic / barrier / L1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyMethodKind {
    /// Fixed penalty coefficient — never updated.
    /// Suitable when a good penalty value is known in advance.
    Static,

    /// Penalty is multiplied by a constant factor each outer iteration.
    /// Classic "increasing penalty" approach for exterior methods.
    Dynamic,

    /// Penalty is updated based on constraint violation progress.
    /// Increases faster when violations are not decreasing, avoids ill-
    /// conditioning by capping growth when convergence is progressing well.
    Adaptive,

    /// Augmented Lagrangian — maintains Lagrange multiplier estimates and
    /// updates them after each outer iteration via dual ascent.  The penalty
    /// parameter increases only when the multiplier update stalls.
    AugmentedLagrangian,
}

// ─────────────────────────────────────────────────────────────────────────────
// penalty_function — standalone evaluation helper
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the penalised objective at point `x`.
///
/// For *exterior* (quadratic) penalties:
/// ```text
/// P(x, mu) = f(x)
///          + mu * Σ_i max(0, g_i(x))²     [inequality: g_i(x) <= 0]
///          + mu * Σ_j h_j(x)²              [equality:   h_j(x)  = 0]
/// ```
///
/// For *interior* (log-barrier) penalties:
/// ```text
/// B(x, mu) = f(x) - mu * Σ_i ln(-g_i(x))     [only for strictly feasible x]
/// ```
///
/// # Arguments
/// * `x`              - Current iterate (decision vector).
/// * `obj`            - Objective function `f: &[f64] -> f64`.
/// * `ineq_cons`      - Inequality constraints `g_i(x) <= 0`.
/// * `eq_cons`        - Equality constraints `h_j(x) = 0`.
/// * `penalty_coeff`  - Scalar penalty / barrier weight `μ`.
/// * `kind`           - Which form of penalty to apply.
///
/// # Returns
/// Penalised scalar value.  Returns `f64::INFINITY` if the log-barrier is
/// requested but `x` is infeasible (any `g_i(x) >= 0`).
pub fn penalty_function<F, G, H>(
    x: &[f64],
    obj: F,
    ineq_cons: &[G],
    eq_cons: &[H],
    penalty_coeff: f64,
    kind: PenaltyKind,
) -> f64
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> f64,
    H: Fn(&[f64]) -> f64,
{
    let f_val = obj(x);
    match kind {
        PenaltyKind::External => {
            let ineq_pen: f64 = ineq_cons
                .iter()
                .map(|g| penalty_coeff * g(x).max(0.0).powi(2))
                .sum();
            let eq_pen: f64 = eq_cons.iter().map(|h| penalty_coeff * h(x).powi(2)).sum();
            f_val + ineq_pen + eq_pen
        }
        PenaltyKind::Interior => {
            let barrier: f64 = ineq_cons
                .iter()
                .map(|g| {
                    let gv = g(x);
                    if gv < -1e-15 {
                        -penalty_coeff * gv.abs().ln()
                    } else {
                        f64::INFINITY
                    }
                })
                .sum();
            f_val + barrier
        }
        PenaltyKind::ExactL1 => {
            let ineq_pen: f64 = ineq_cons
                .iter()
                .map(|g| penalty_coeff * g(x).max(0.0))
                .sum();
            let eq_pen: f64 = eq_cons.iter().map(|h| penalty_coeff * h(x).abs()).sum();
            f_val + ineq_pen + eq_pen
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptivePenalty
// ─────────────────────────────────────────────────────────────────────────────

/// Adaptive penalty controller.
///
/// Tracks constraint violation history and dynamically adjusts the penalty
/// coefficient:
/// - If violation has not improved by `improvement_threshold` fraction compared
///   to the previous iteration → multiply penalty by `increase_factor`.
/// - If violation improved substantially and current penalty is high → allow
///   mild reduction to avoid ill-conditioning (optional, controlled by
///   `allow_decrease`).
///
/// This implements a simplified version of the adaptive penalty from:
/// Farmani & Wright (2003), "Self-adaptive fitness formulation for constrained
/// optimization", IEEE TEC 7(5):445-455.
#[derive(Debug, Clone)]
pub struct AdaptivePenalty {
    /// Current penalty coefficient.
    pub penalty_coeff: f64,
    /// Minimum allowed penalty (lower bound on growth).
    pub min_penalty: f64,
    /// Maximum allowed penalty (avoids numerical ill-conditioning).
    pub max_penalty: f64,
    /// Multiplicative increase factor applied when violations stall.
    pub increase_factor: f64,
    /// Multiplicative decrease factor applied when violations decrease rapidly.
    pub decrease_factor: f64,
    /// Relative improvement threshold below which penalty is increased.
    /// E.g., 0.1 means "less than 10% improvement triggers increase".
    pub improvement_threshold: f64,
    /// Whether to allow penalty *decreases* (can reduce ill-conditioning).
    pub allow_decrease: bool,
    /// Stored constraint violation from previous outer iteration.
    prev_violation: f64,
    /// Number of consecutive non-improving iterations.
    stall_count: usize,
    /// Stall patience: how many non-improving iters before increasing penalty.
    pub stall_patience: usize,
}

impl Default for AdaptivePenalty {
    fn default() -> Self {
        AdaptivePenalty {
            penalty_coeff: 1.0,
            min_penalty: 1e-3,
            max_penalty: 1e10,
            increase_factor: 10.0,
            decrease_factor: 0.5,
            improvement_threshold: 0.25,
            allow_decrease: false,
            prev_violation: f64::INFINITY,
            stall_count: 0,
            stall_patience: 1,
        }
    }
}

impl AdaptivePenalty {
    /// Create a new adaptive penalty controller with the given initial coefficient.
    pub fn new(initial_penalty: f64) -> Self {
        AdaptivePenalty {
            penalty_coeff: initial_penalty,
            ..Default::default()
        }
    }

    /// Create with full configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        initial_penalty: f64,
        min_penalty: f64,
        max_penalty: f64,
        increase_factor: f64,
        decrease_factor: f64,
        improvement_threshold: f64,
        allow_decrease: bool,
        stall_patience: usize,
    ) -> Self {
        AdaptivePenalty {
            penalty_coeff: initial_penalty,
            min_penalty,
            max_penalty,
            increase_factor,
            decrease_factor,
            improvement_threshold,
            allow_decrease,
            prev_violation: f64::INFINITY,
            stall_count: 0,
            stall_patience,
        }
    }

    /// Update the penalty coefficient based on `current_violation`.
    ///
    /// Should be called once per outer iteration after the sub-problem has been
    /// solved.  Returns the updated penalty coefficient.
    pub fn update(&mut self, current_violation: f64) -> f64 {
        if self.prev_violation.is_infinite() {
            // First call — just record and return current value
            self.prev_violation = current_violation;
            return self.penalty_coeff;
        }

        let relative_improvement = if self.prev_violation > 1e-15 {
            (self.prev_violation - current_violation) / self.prev_violation
        } else {
            // Already near zero — treat as converged
            1.0
        };

        if relative_improvement < self.improvement_threshold {
            // Not improving fast enough
            self.stall_count += 1;
            if self.stall_count >= self.stall_patience {
                self.penalty_coeff =
                    (self.penalty_coeff * self.increase_factor).min(self.max_penalty);
                self.stall_count = 0;
            }
        } else {
            // Good improvement
            self.stall_count = 0;
            if self.allow_decrease && relative_improvement > 0.5 {
                self.penalty_coeff =
                    (self.penalty_coeff * self.decrease_factor).max(self.min_penalty);
            }
        }

        self.prev_violation = current_violation;
        self.penalty_coeff
    }

    /// Reset internal state (violation history and stall counter).
    pub fn reset(&mut self) {
        self.prev_violation = f64::INFINITY;
        self.stall_count = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AugmentedLagrangianSolver
// ─────────────────────────────────────────────────────────────────────────────

/// Options for [`AugmentedLagrangianSolver`].
#[derive(Debug, Clone)]
pub struct AugLagOptions {
    /// Initial penalty parameter ρ.
    pub rho_init: f64,
    /// Maximum penalty parameter.
    pub rho_max: f64,
    /// Factor by which ρ is multiplied when the constraint violation does not
    /// decrease sufficiently.
    pub rho_factor: f64,
    /// Maximum number of outer (multiplier update) iterations.
    pub max_outer_iter: usize,
    /// Constraint violation tolerance (outer loop convergence criterion).
    pub constraint_tol: f64,
    /// Optimality tolerance passed to the inner unconstrained sub-solver.
    pub optimality_tol: f64,
    /// Finite-difference step for inner gradient computation.
    pub eps: f64,
    /// Required relative reduction in violation before multipliers are updated
    /// (otherwise only the penalty grows).
    pub violation_reduction_threshold: f64,
}

impl Default for AugLagOptions {
    fn default() -> Self {
        AugLagOptions {
            rho_init: 1.0,
            rho_max: 1e10,
            rho_factor: 10.0,
            max_outer_iter: 100,
            constraint_tol: 1e-6,
            optimality_tol: 1e-8,
            eps: 1e-7,
            violation_reduction_threshold: 0.25,
        }
    }
}

/// Result from [`AugmentedLagrangianSolver`].
#[derive(Debug, Clone)]
pub struct AugLagResult {
    /// Optimal decision vector.
    pub x: Array1<f64>,
    /// Objective value at x.
    pub fun: f64,
    /// Number of outer iterations performed.
    pub nit: usize,
    /// Total function evaluations (inner + outer).
    pub nfev: usize,
    /// Success flag.
    pub success: bool,
    /// Status message.
    pub message: String,
    /// Final Lagrange multipliers for equality constraints.
    pub lambda_eq: Vec<f64>,
    /// Final Lagrange multipliers for inequality constraints.
    pub lambda_ineq: Vec<f64>,
    /// Final penalty parameter.
    pub rho: f64,
    /// Final constraint violation.
    pub constraint_violation: f64,
}

impl From<AugLagResult> for OptimizeResults<f64> {
    fn from(r: AugLagResult) -> Self {
        OptimizeResults {
            x: r.x,
            fun: r.fun,
            jac: None,
            hess: None,
            constr: None,
            nit: r.nit,
            nfev: r.nfev,
            njev: 0,
            nhev: 0,
            maxcv: 0,
            message: r.message,
            success: r.success,
            status: if r.success { 0 } else { 1 },
        }
    }
}

/// Augmented Lagrangian method solver (Method of Multipliers).
///
/// Solves problems of the form:
/// ```text
/// min   f(x)
/// s.t.  h_j(x) = 0      (equality)
///       g_i(x) <= 0     (inequality)
/// ```
///
/// The augmented Lagrangian for equality constraints is:
/// ```text
/// L_A(x, λ, ρ) = f(x) + Σ_j λ_j h_j(x) + (ρ/2) Σ_j h_j(x)²
/// ```
/// Inequality constraints are handled via the shifted/signed penalty
/// (Rockafellar's form):
/// ```text
/// L_A += Σ_i [ λ_i g_i(x) + (ρ/2) g_i(x)² ]   when  g_i(x) + λ_i/ρ > 0
///      + 0                                        otherwise
/// ```
///
/// Multipliers are updated each outer iteration:
/// ```text
/// λ_j ← λ_j + ρ h_j(x*)     (equality)
/// λ_i ← max(0, λ_i + ρ g_i(x*))   (inequality)
/// ```
///
/// # References
/// - Nocedal & Wright (2006), §17.4, "Augmented Lagrangian Methods".
/// - Bertsekas (1982), "Constrained Optimization and Lagrange Multiplier Methods".
#[derive(Debug, Clone)]
pub struct AugmentedLagrangianSolver {
    /// Solver configuration.
    pub options: AugLagOptions,
}

impl AugmentedLagrangianSolver {
    /// Create with default options.
    pub fn new() -> Self {
        AugmentedLagrangianSolver {
            options: AugLagOptions::default(),
        }
    }

    /// Create with custom options.
    pub fn with_options(options: AugLagOptions) -> Self {
        AugmentedLagrangianSolver { options }
    }

    /// Compute total constraint violation (L2 norm of violations).
    fn compute_violation<E, G>(&self, x: &[f64], eq_cons: &[E], ineq_cons: &[G]) -> f64
    where
        E: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        let eq_sq: f64 = eq_cons.iter().map(|h| h(x).powi(2)).sum();
        let ineq_sq: f64 = ineq_cons.iter().map(|g| g(x).max(0.0).powi(2)).sum();
        (eq_sq + ineq_sq).sqrt()
    }

    /// Solve the augmented Lagrangian problem.
    ///
    /// # Arguments
    /// * `f`        - Objective function.
    /// * `eq_cons`  - Equality constraints `h_j(x) = 0`.
    /// * `ineq_cons`- Inequality constraints `g_i(x) <= 0`.
    /// * `x0`       - Initial iterate.
    pub fn solve<F, E, G>(
        &self,
        f: F,
        eq_cons: &[E],
        ineq_cons: &[G],
        x0: &[f64],
    ) -> OptimizeResult<AugLagResult>
    where
        F: Fn(&[f64]) -> f64,
        E: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        let n_eq = eq_cons.len();
        let n_ineq = ineq_cons.len();

        // Initialise multipliers at zero
        let mut lambda_eq = vec![0.0_f64; n_eq];
        let mut lambda_ineq = vec![0.0_f64; n_ineq];
        let mut rho = self.options.rho_init;

        let mut x = x0.to_vec();
        let mut nfev_total = 0usize;
        let mut nit = 0usize;
        let mut prev_violation = f64::INFINITY;

        for _outer in 0..self.options.max_outer_iter {
            nit += 1;

            // Snapshot multipliers and penalty for the closure
            let lam_eq_snap = lambda_eq.clone();
            let lam_ineq_snap = lambda_ineq.clone();
            let rho_snap = rho;

            // Augmented Lagrangian function
            let aug_lag = |xv: &[f64]| -> f64 {
                let mut val = f(xv);

                // Equality terms: λ_j h_j + (ρ/2) h_j²
                for (j, h) in eq_cons.iter().enumerate() {
                    let hv = h(xv);
                    val += lam_eq_snap[j] * hv + 0.5 * rho_snap * hv * hv;
                }

                // Inequality terms (Rockafellar shifted form)
                // If  g_i + λ_i/ρ > 0  (active or violated):
                //   contribution = λ_i g_i + (ρ/2) g_i²
                //               = (1/(2ρ)) [ (λ_i + ρ g_i)² - λ_i² ]
                // If  g_i + λ_i/ρ <= 0 (inactive, well inside feasible region):
                //   contribution = -λ_i²/(2ρ)  [constant w.r.t. x, omit]
                for (i, g) in ineq_cons.iter().enumerate() {
                    let gv = g(xv);
                    let shifted = gv + lam_ineq_snap[i] / rho_snap;
                    if shifted > 0.0 {
                        val += lam_ineq_snap[i] * gv + 0.5 * rho_snap * gv * gv;
                    }
                    // else: constraint is inactive, no contribution
                }

                val
            };

            // Minimise the augmented Lagrangian (unconstrained sub-problem)
            let new_x = minimize_penalty_subproblem(
                aug_lag,
                &x,
                2000,
                self.options.optimality_tol,
                self.options.eps,
                &mut nfev_total,
            );

            x = new_x;

            // Compute current violation
            let cv = self.compute_violation(&x, eq_cons, ineq_cons);

            // Check for convergence
            if cv <= self.options.constraint_tol {
                let fun = f(&x);
                return Ok(AugLagResult {
                    x: Array1::from_vec(x),
                    fun,
                    nit,
                    nfev: nfev_total,
                    success: true,
                    message: "Converged: constraint violation below tolerance".to_string(),
                    lambda_eq,
                    lambda_ineq,
                    rho,
                    constraint_violation: cv,
                });
            }

            // Determine whether to update multipliers or just increase penalty
            let violation_improvement = if prev_violation.is_finite() && prev_violation > 1e-15 {
                (prev_violation - cv) / prev_violation
            } else {
                0.0
            };

            if violation_improvement >= self.options.violation_reduction_threshold {
                // Good progress: update multipliers (dual ascent step)
                for (j, h) in eq_cons.iter().enumerate() {
                    lambda_eq[j] += rho * h(&x);
                }
                for (i, g) in ineq_cons.iter().enumerate() {
                    lambda_ineq[i] = (lambda_ineq[i] + rho * g(&x)).max(0.0);
                }
            } else {
                // Poor progress: increase penalty to force constraint satisfaction
                rho = (rho * self.options.rho_factor).min(self.options.rho_max);
            }

            prev_violation = cv;
        }

        // Max iterations reached
        let cv = self.compute_violation(&x, eq_cons, ineq_cons);
        let fun = f(&x);
        let success = cv <= self.options.constraint_tol;
        Ok(AugLagResult {
            x: Array1::from_vec(x),
            fun,
            nit,
            nfev: nfev_total,
            success,
            message: if success {
                "Converged".to_string()
            } else {
                format!(
                    "Maximum outer iterations ({}) reached; cv={:.2e}",
                    self.options.max_outer_iter, cv
                )
            },
            lambda_eq,
            lambda_ineq,
            rho,
            constraint_violation: cv,
        })
    }
}

impl Default for AugmentedLagrangianSolver {
    fn default() -> Self {
        AugmentedLagrangianSolver::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_penalty_equality_constraint() {
        // min x^2 + y^2  s.t. x + y = 1
        // Solution: x = y = 0.5, f = 0.5
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let h = |x: &[f64]| x[0] + x[1] - 1.0;

        let opts = PenaltyOptions {
            kind: PenaltyKind::External,
            mu_init: 1.0,
            mu_factor: 10.0,
            mu_max: 1e8,
            max_outer_iter: 30,
            constraint_tol: 1e-4,
            ..Default::default()
        };
        let solver = PenaltyMethod::with_options(opts);
        let result = solver
            .solve_slice(f, &[h], &[] as &[fn(&[f64]) -> f64], &[0.0, 0.0])
            .expect("solve failed");

        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[1], 0.5, epsilon = 1e-2);
        assert_abs_diff_eq!(result.fun, 0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_penalty_inequality_constraint() {
        // min x^2 + y^2  s.t. x + y >= 1  (g: 1 - x - y <= 0)
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let g = |x: &[f64]| 1.0 - x[0] - x[1]; // <= 0 means x+y >= 1

        let opts = PenaltyOptions {
            kind: PenaltyKind::External,
            mu_init: 1.0,
            mu_factor: 10.0,
            mu_max: 1e8,
            max_outer_iter: 40,
            constraint_tol: 1e-3,
            ..Default::default()
        };
        let solver = PenaltyMethod::with_options(opts);
        let result = solver
            .solve_slice(f, &[] as &[fn(&[f64]) -> f64], &[g], &[2.0, 2.0])
            .expect("solve failed");

        // Solution: (0.5, 0.5)
        assert_abs_diff_eq!(result.fun, 0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_penalty_no_constraints() {
        // min (x-3)^2 + (y-4)^2 unconstrained
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2);
        let solver = PenaltyMethod::new();
        let result = solver
            .solve_slice(
                f,
                &[] as &[fn(&[f64]) -> f64],
                &[] as &[fn(&[f64]) -> f64],
                &[0.0, 0.0],
            )
            .expect("solve failed");

        assert_abs_diff_eq!(result.fun, 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[0], 3.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[1], 4.0, epsilon = 1e-2);
    }

    #[test]
    fn test_penalty_l1_equality() {
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let h = |x: &[f64]| x[0] + x[1] - 3.0;

        let opts = PenaltyOptions {
            kind: PenaltyKind::ExactL1,
            mu_init: 10.0,
            max_outer_iter: 50,
            constraint_tol: 1e-3,
            ..Default::default()
        };
        let solver = PenaltyMethod::with_options(opts);
        let result = solver
            .solve_slice(f, &[h], &[] as &[fn(&[f64]) -> f64], &[0.0, 0.0])
            .expect("solve failed");

        // Solution: (1, 2) satisfies x+y=3
        assert_abs_diff_eq!(result.fun, 0.0, epsilon = 1e-2);
    }

    #[test]
    fn test_penalty_interior_barrier() {
        // min (x-0.5)^2  s.t. x < 1 (g: x - 0.999 <= 0)
        // Start strictly inside: x0 = 0.3
        let f = |x: &[f64]| (x[0] - 0.5).powi(2);
        let g = |x: &[f64]| x[0] - 0.999;
        let opts = PenaltyOptions {
            kind: PenaltyKind::Interior,
            mu_init: 0.1,
            mu_factor: 5.0,
            max_outer_iter: 20,
            ..Default::default()
        };
        let solver = PenaltyMethod::with_options(opts);
        let result = solver
            .solve_slice(f, &[] as &[fn(&[f64]) -> f64], &[g], &[0.3])
            .expect("solve failed");

        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_penalty_mixed_constraints() {
        // min x^2 + y^2 + z^2  s.t. x+y+z=3, z<=1
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2) + x[2].powi(2);
        let h = |x: &[f64]| x[0] + x[1] + x[2] - 3.0;
        let g = |x: &[f64]| x[2] - 1.0;

        let opts = PenaltyOptions {
            kind: PenaltyKind::External,
            mu_init: 1.0,
            mu_factor: 5.0,
            mu_max: 1e7,
            max_outer_iter: 50,
            constraint_tol: 1e-3,
            ..Default::default()
        };
        let solver = PenaltyMethod::with_options(opts);
        let result = solver
            .solve_slice(f, &[h], &[g], &[0.0, 0.0, 0.0])
            .expect("solve failed");

        // With z<=1 and x+y+z=3: optimal at z=1, x=y=1 -> f=3
        assert!(result.fun <= 4.0, "fun={}", result.fun);
    }

    #[test]
    fn test_penalty_arrayview1_interface() {
        use scirs2_core::ndarray::{array, ArrayView1};
        let f = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let h = |x: &ArrayView1<f64>| x[0] + x[1] - 1.0;

        let opts = PenaltyOptions {
            kind: PenaltyKind::External,
            max_outer_iter: 20,
            constraint_tol: 1e-3,
            ..Default::default()
        };
        let solver = PenaltyMethod::with_options(opts);
        let x0 = array![0.0, 0.0];
        let result = solver
            .solve(f, &[h], &[] as &[fn(&ArrayView1<f64>) -> f64], &x0)
            .expect("solve failed");

        // Solution: (0.5, 0.5)
        assert_abs_diff_eq!(result.fun, 0.5, epsilon = 5e-2);
    }

    // ── penalty_function free function ───────────────────────────────────────

    #[test]
    fn test_penalty_function_external_feasible() {
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let g = |x: &[f64]| x[0] + x[1] - 1.0; // g <= 0 means x+y <= 1; at (0,0) g = -1 <= 0

        // At feasible point (0,0): g(x) = -1 < 0, no violation
        let val = penalty_function(
            &[0.0, 0.0],
            f,
            &[g],
            &[] as &[fn(&[f64]) -> f64],
            10.0,
            PenaltyKind::External,
        );
        // No penalty applied: val = f(0,0) = 0
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_penalty_function_external_violated() {
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let g = |x: &[f64]| x[0] + x[1] - 1.0; // at (1,1): g = 1 > 0, violated

        let mu = 5.0;
        let val = penalty_function(
            &[1.0, 1.0],
            f,
            &[g],
            &[] as &[fn(&[f64]) -> f64],
            mu,
            PenaltyKind::External,
        );
        // f = 2, penalty = mu * max(0, 1)^2 = 5
        assert_abs_diff_eq!(val, 2.0 + mu * 1.0_f64.powi(2), epsilon = 1e-12);
    }

    #[test]
    fn test_penalty_function_equality() {
        let f = |_x: &[f64]| 0.0;
        let h = |x: &[f64]| x[0] - 1.0; // h = 0 at x=1; at x=2 h=1

        let mu = 3.0;
        let val = penalty_function(
            &[2.0],
            f,
            &[] as &[fn(&[f64]) -> f64],
            &[h],
            mu,
            PenaltyKind::External,
        );
        // penalty = mu * h^2 = 3 * 1 = 3
        assert_abs_diff_eq!(val, 3.0, epsilon = 1e-12);
    }

    // ── AdaptivePenalty ──────────────────────────────────────────────────────

    #[test]
    fn test_adaptive_penalty_increases_on_stall() {
        let mut ap = AdaptivePenalty::new(1.0);
        ap.increase_factor = 5.0;
        ap.improvement_threshold = 0.1;
        ap.stall_patience = 1;

        // First call: records violation, returns current penalty
        let p0 = ap.update(1.0);
        assert_abs_diff_eq!(p0, 1.0, epsilon = 1e-12);

        // Second call: violation is 0.95 (only 5% improvement < threshold 10%)
        // => stall_count=1 >= patience=1 => penalty should increase
        let p1 = ap.update(0.95);
        assert!(p1 > 1.0, "Penalty should have increased; got {p1}");
    }

    #[test]
    fn test_adaptive_penalty_no_increase_on_good_progress() {
        let mut ap = AdaptivePenalty::new(1.0);
        ap.improvement_threshold = 0.1;
        ap.stall_patience = 1;

        ap.update(1.0); // seed prev_violation
        let p1 = ap.update(0.5); // 50% improvement, well above 10% threshold
                                 // Penalty should NOT increase (allow_decrease=false by default)
        assert_abs_diff_eq!(p1, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_adaptive_penalty_capped_at_max() {
        let mut ap = AdaptivePenalty::with_config(1e9, 1e-3, 1e10, 1000.0, 0.5, 0.1, false, 1);
        ap.update(1.0);
        let p = ap.update(1.0); // no improvement → multiply by 1000 → capped at 1e10
        assert!(p <= 1e10, "Penalty exceeded max; got {p}");
    }

    // ── AugmentedLagrangianSolver ────────────────────────────────────────────

    #[test]
    fn test_aug_lag_equality_constraint() {
        // min x^2 + y^2  s.t. x + y = 1
        // Solution: x = y = 0.5, f = 0.5
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let h = |x: &[f64]| x[0] + x[1] - 1.0;

        let opts = AugLagOptions {
            rho_init: 1.0,
            rho_factor: 5.0,
            max_outer_iter: 50,
            constraint_tol: 1e-4,
            ..Default::default()
        };
        let solver = AugmentedLagrangianSolver::with_options(opts);
        let result = solver
            .solve(f, &[h], &[] as &[fn(&[f64]) -> f64], &[0.0, 0.0])
            .expect("AugLag solve failed");

        assert!(
            result.success || result.constraint_violation < 1e-2,
            "cv={}",
            result.constraint_violation
        );
        assert_abs_diff_eq!(result.fun, 0.5, epsilon = 0.05);
    }

    #[test]
    fn test_aug_lag_inequality_constraint() {
        // min x^2  s.t. x >= 1  =>  g(x) = 1 - x <= 0
        // Solution: x = 1
        let f = |x: &[f64]| x[0].powi(2);
        let g = |x: &[f64]| 1.0 - x[0]; // <= 0 means x >= 1

        let opts = AugLagOptions {
            rho_init: 1.0,
            rho_factor: 10.0,
            max_outer_iter: 60,
            constraint_tol: 1e-3,
            ..Default::default()
        };
        let solver = AugmentedLagrangianSolver::with_options(opts);
        let result = solver
            .solve(f, &[] as &[fn(&[f64]) -> f64], &[g], &[0.5])
            .expect("AugLag solve failed");

        // x should be near 1.0
        assert!(
            result.x[0] >= 0.9 && result.x[0] <= 1.2,
            "Expected x~1.0, got x={}",
            result.x[0]
        );
    }

    #[test]
    fn test_aug_lag_no_constraints() {
        // Unconstrained: min (x-2)^2 + (y-3)^2
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
        let solver = AugmentedLagrangianSolver::new();
        let result = solver
            .solve(
                f,
                &[] as &[fn(&[f64]) -> f64],
                &[] as &[fn(&[f64]) -> f64],
                &[0.0, 0.0],
            )
            .expect("AugLag solve failed");

        assert_abs_diff_eq!(result.x[0], 2.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.x[1], 3.0, epsilon = 1e-2);
        assert_abs_diff_eq!(result.fun, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_penalty_method_kind_variants() {
        // Just verify the enum can be constructed and compared
        assert_eq!(PenaltyMethodKind::Static, PenaltyMethodKind::Static);
        assert_ne!(PenaltyMethodKind::Dynamic, PenaltyMethodKind::Adaptive);
        assert_ne!(
            PenaltyMethodKind::AugmentedLagrangian,
            PenaltyMethodKind::Static
        );
    }
}
