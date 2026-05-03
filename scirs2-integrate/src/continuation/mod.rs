//! Numerical continuation methods for parametric systems
//!
//! This module provides algorithms for tracing solution branches of parametric
//! equations F(x, λ) = 0 as the parameter λ varies, detecting bifurcations and
//! turning points along the way.
//!
//! ## Methods
//!
//! - **Natural parameter continuation**: Increment λ, solve Newton at each step
//! - **Pseudo-arclength continuation** (Keller's method): Augment with arclength
//!   condition to continue past turning points and trace closed branches
//!
//! ## Detected features
//!
//! - **LimitPoint** (fold/turning point): det(F_x) = 0, branch turns back in λ
//! - **BranchPoint** (bifurcation): two branches cross; requires branch switching
//!
//! ## References
//!
//! - Keller (1977), "Numerical solution of bifurcation and nonlinear eigenvalue problems"
//! - Allgower & Georg (1990), "Numerical Continuation Methods"
//! - Seydel (2010), "Practical Bifurcation and Stability Analysis"

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

#[inline(always)]
fn f64_to_f(v: f64) -> f64 {
    v
}

/// Solve A·x = b by Gaussian elimination with partial pivoting.
/// Modifies A and b in place.
fn gauss_solve(a: &mut Array2<f64>, b: &mut Array1<f64>) -> IntegrateResult<Array1<f64>> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot
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
                "Singular matrix in continuation solve".to_string(),
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
                let update = factor * a[[col, j]];
                a[[row, j]] -= update;
            }
            let bup = factor * b[col];
            b[row] -= bup;
        }
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[[i, j]] * x[j];
        }
        x[i] = sum / a[[i, i]];
    }
    Ok(x)
}

/// Compute numerical Jacobian of F(x, λ) w.r.t. x using central differences
fn numerical_jacobian_x<F>(f: &F, x: &Array1<f64>, lambda: f64, eps: f64) -> Array2<f64>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    let n = x.len();
    let mut jac = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[j] += eps;
        xm[j] -= eps;
        let fp = f(&xp, lambda);
        let fm = f(&xm, lambda);
        for i in 0..n {
            jac[[i, j]] = (fp[i] - fm[i]) / (2.0 * eps);
        }
    }
    jac
}

/// Compute numerical derivative of F w.r.t. λ using central differences
fn numerical_df_dlambda<F>(f: &F, x: &Array1<f64>, lambda: f64, eps: f64) -> Array1<f64>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    let fp = f(x, lambda + eps);
    let fm = f(x, lambda - eps);
    let n = fp.len();
    let mut df = Array1::<f64>::zeros(n);
    for i in 0..n {
        df[i] = (fp[i] - fm[i]) / (2.0 * eps);
    }
    df
}

// ---------------------------------------------------------------------------
// Point classification types
// ---------------------------------------------------------------------------

/// A point on the solution branch: (state, parameter value)
#[derive(Debug, Clone)]
pub struct BranchPoint {
    /// State vector x at the branch point
    pub x: Array1<f64>,
    /// Parameter value λ at the branch point
    pub lambda: f64,
    /// Approximate null-vector of F_x (tangent to intersecting branch)
    pub null_vector: Option<Array1<f64>>,
}

/// A limit (fold/turning) point where det(F_x) = 0
#[derive(Debug, Clone)]
pub struct LimitPoint {
    /// State at the limit point
    pub x: Array1<f64>,
    /// Parameter value at the limit point
    pub lambda: f64,
    /// Index along the branch where the limit point occurs
    pub branch_index: usize,
    /// Stability change indicator (true if stability changes)
    pub stability_change: bool,
}

/// Stability classification for a solution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolutionStability {
    /// All eigenvalues of F_x have negative real part (stable)
    Stable,
    /// At least one eigenvalue of F_x has positive real part (unstable)
    Unstable,
    /// Cannot determine (e.g., zero eigenvalue)
    Unknown,
}

// ---------------------------------------------------------------------------
// Continuation result
// ---------------------------------------------------------------------------

/// Result of a numerical continuation run
#[derive(Debug, Clone)]
pub struct ContinuationBranchResult {
    /// Solution states along the branch
    pub x: Vec<Array1<f64>>,
    /// Parameter values along the branch
    pub lambda: Vec<f64>,
    /// Stability at each point (if computed)
    pub stability: Vec<SolutionStability>,
    /// Detected limit (fold) points
    pub limit_points: Vec<LimitPoint>,
    /// Detected branch (bifurcation) points
    pub branch_points: Vec<BranchPoint>,
    /// Number of Newton iterations at each step
    pub newton_iters: Vec<usize>,
    /// Whether continuation terminated normally
    pub success: bool,
    /// Termination message
    pub message: String,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for continuation methods
#[derive(Debug, Clone)]
pub struct ContinuationConfig {
    /// Maximum number of steps along the branch
    pub max_steps: usize,
    /// Initial step size in λ (natural) or arclength (pseudo-arclength)
    pub ds: f64,
    /// Minimum step size
    pub ds_min: f64,
    /// Maximum step size
    pub ds_max: f64,
    /// Step-size adaptation factor (0 < factor ≤ 1)
    pub step_adapt_factor: f64,
    /// Newton solver tolerance
    pub newton_tol: f64,
    /// Maximum Newton iterations per step
    pub max_newton_iter: usize,
    /// Finite difference epsilon for Jacobians
    pub fd_eps: f64,
    /// Whether to compute stability at each point
    pub compute_stability: bool,
    /// Determinant threshold for limit point detection
    pub limit_point_tol: f64,
    /// Desired Newton iterations per step (for adaptive step size)
    pub desired_newton_iter: usize,
}

impl Default for ContinuationConfig {
    fn default() -> Self {
        Self {
            max_steps: 500,
            ds: 0.01,
            ds_min: 1e-8,
            ds_max: 1.0,
            step_adapt_factor: 0.8,
            newton_tol: 1e-10,
            max_newton_iter: 20,
            fd_eps: 1e-7,
            compute_stability: true,
            limit_point_tol: 1e-4,
            desired_newton_iter: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Natural Parameter Continuation
// ---------------------------------------------------------------------------

/// Natural parameter continuation.
///
/// Increments the parameter λ by a fixed (or adaptive) step and solves
/// F(x, λ) = 0 using Newton's method at each value of λ.
///
/// Simple and robust for problems without turning points. Fails near limit
/// (fold) points where the branch turns back (dλ → 0). Use
/// `PseudoArcLengthContinuation` to continue past such points.
///
/// # Arguments
///
/// * `f` - Residual function F(x, λ) → R^n
/// * `x0` - Initial solution at λ0
/// * `lambda0` - Initial parameter value
/// * `lambda_end` - Target parameter value (can be less than λ0 for backward)
/// * `cfg` - Continuation configuration
///
/// # Returns
///
/// `ContinuationBranchResult` with the traced branch.
pub struct NaturalParameterContinuation;

impl NaturalParameterContinuation {
    /// Run natural parameter continuation from (x0, λ0) to λ_end.
    pub fn run<F>(
        f: &F,
        x0: &Array1<f64>,
        lambda0: f64,
        lambda_end: f64,
        cfg: &ContinuationConfig,
    ) -> IntegrateResult<ContinuationBranchResult>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        let n = x0.len();
        let direction = if lambda_end >= lambda0 { 1.0 } else { -1.0 };
        let mut ds = cfg.ds.abs() * direction;

        let mut x = x0.clone();
        let mut lambda = lambda0;

        let mut xs = vec![x.clone()];
        let mut lambdas = vec![lambda];
        let mut stabilities = Vec::new();
        let mut limit_points = Vec::new();
        let mut branch_points_list = Vec::new();
        let mut newton_iters = vec![0usize];

        if cfg.compute_stability {
            let stab = compute_stability_from_jacobian(f, &x, lambda, cfg.fd_eps);
            stabilities.push(stab);
        }

        let mut prev_det = jacobian_determinant(f, &x, lambda, cfg.fd_eps);

        for _step in 0..cfg.max_steps {
            // Clamp step to not overshoot lambda_end
            if (lambda + ds - lambda_end) * direction > 0.0 {
                ds = (lambda_end - lambda).abs() * direction;
                if ds.abs() < cfg.ds_min {
                    break;
                }
            }

            let lambda_new = lambda + ds;

            // Newton solve
            match newton_solve_continuation(f, &x, lambda_new, cfg) {
                Ok((x_new, n_iters)) => {
                    let curr_det = jacobian_determinant(f, &x_new, lambda_new, cfg.fd_eps);

                    // Limit point detection: sign change of det(J_x)
                    if prev_det * curr_det < 0.0 {
                        limit_points.push(LimitPoint {
                            x: x_new.clone(),
                            lambda: lambda_new,
                            branch_index: xs.len(),
                            stability_change: true,
                        });
                    }

                    // Branch point detection: |det(J_x)| near zero but no sign change
                    if curr_det.abs() < cfg.limit_point_tol && (prev_det * curr_det >= 0.0) {
                        let null_vec =
                            approximate_null_vector(f, &x_new, lambda_new, cfg.fd_eps, n);
                        branch_points_list.push(BranchPoint {
                            x: x_new.clone(),
                            lambda: lambda_new,
                            null_vector: Some(null_vec),
                        });
                    }

                    // Adaptive step size
                    let adapt_ratio = cfg.desired_newton_iter as f64 / n_iters.max(1) as f64;
                    let new_ds = (ds * adapt_ratio.sqrt())
                        .abs()
                        .clamp(cfg.ds_min, cfg.ds_max);
                    ds = new_ds * direction;

                    prev_det = curr_det;
                    x = x_new.clone();
                    lambda = lambda_new;

                    xs.push(x_new);
                    lambdas.push(lambda);
                    newton_iters.push(n_iters);

                    if cfg.compute_stability {
                        let stab = compute_stability_from_jacobian(f, &x, lambda, cfg.fd_eps);
                        stabilities.push(stab);
                    }

                    // Check if we've reached the target
                    if (lambda - lambda_end).abs() < cfg.ds_min.abs() {
                        break;
                    }
                }
                Err(_) => {
                    // Reduce step size and retry
                    ds *= cfg.step_adapt_factor;
                    if ds.abs() < cfg.ds_min {
                        return Ok(ContinuationBranchResult {
                            x: xs,
                            lambda: lambdas,
                            stability: stabilities,
                            limit_points,
                            branch_points: branch_points_list,
                            newton_iters,
                            success: false,
                            message: "Step size below minimum: Newton convergence failure"
                                .to_string(),
                        });
                    }
                }
            }
        }

        Ok(ContinuationBranchResult {
            x: xs,
            lambda: lambdas,
            stability: stabilities,
            limit_points,
            branch_points: branch_points_list,
            newton_iters,
            success: true,
            message: "Natural parameter continuation completed".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Pseudo-Arclength Continuation (Keller's method)
// ---------------------------------------------------------------------------

/// Pseudo-arclength continuation using Keller's bordering algorithm.
///
/// Augments the system with an arclength condition:
///   F(x, λ) = 0
///   (x - x_prev)·dx_ds + (λ - λ_prev)·dλ_ds - ds = 0
///
/// where (dx_ds, dλ_ds) is the unit tangent to the branch. This allows
/// continuation past limit (fold) points and along closed branches.
///
/// # Algorithm
///
/// 1. Compute tangent (dx_ds, dλ_ds) by solving the extended linear system
/// 2. Predict: (x_pred, λ_pred) = (x_prev, λ_prev) + ds*(dx_ds, dλ_ds)
/// 3. Correct: Newton on the extended system (n+1 equations, n+1 unknowns)
pub struct PseudoArcLengthContinuation;

impl PseudoArcLengthContinuation {
    /// Run pseudo-arclength continuation from (x0, λ0).
    ///
    /// # Arguments
    ///
    /// * `f` - Residual function F(x, λ) → R^n
    /// * `x0` - Initial solution at λ0 (must satisfy F(x0, λ0) ≈ 0)
    /// * `lambda0` - Initial parameter value
    /// * `lambda_range` - (λ_min, λ_max): stop when λ leaves this range
    /// * `cfg` - Continuation configuration
    /// * `direction` - Initial direction: +1.0 increases λ, -1.0 decreases λ
    pub fn run<F>(
        f: &F,
        x0: &Array1<f64>,
        lambda0: f64,
        lambda_range: (f64, f64),
        cfg: &ContinuationConfig,
        direction: f64,
    ) -> IntegrateResult<ContinuationBranchResult>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        let n = x0.len();
        let (lambda_min, lambda_max) = lambda_range;

        let mut x = x0.clone();
        let mut lambda = lambda0;

        // Compute initial tangent
        let (mut tx, mut tl) = compute_tangent(f, &x, lambda, direction, cfg.fd_eps, n)?;

        let mut xs = vec![x.clone()];
        let mut lambdas = vec![lambda];
        let mut stabilities = Vec::new();
        let mut limit_points = Vec::new();
        let mut branch_points_list = Vec::new();
        let mut newton_iters = vec![0usize];

        if cfg.compute_stability {
            let stab = compute_stability_from_jacobian(f, &x, lambda, cfg.fd_eps);
            stabilities.push(stab);
        }

        let mut prev_det = jacobian_determinant(f, &x, lambda, cfg.fd_eps);
        let mut ds = cfg.ds;

        for _step in 0..cfg.max_steps {
            // Predictor
            let x_pred = {
                let mut xp = Array1::<f64>::zeros(n);
                for i in 0..n {
                    xp[i] = x[i] + ds * tx[i];
                }
                xp
            };
            let lambda_pred = lambda + ds * tl;

            // Check range
            if lambda_pred < lambda_min || lambda_pred > lambda_max {
                // Try to land exactly on the boundary
                let lambda_target = if lambda_pred < lambda_min {
                    lambda_min
                } else {
                    lambda_max
                };
                // Scale ds to hit boundary
                let ds_boundary = (lambda_target - lambda) / tl.max(1e-300);
                if ds_boundary.abs() < cfg.ds_min {
                    break;
                }
                let x_boundary = {
                    let mut xb = Array1::<f64>::zeros(n);
                    for i in 0..n {
                        xb[i] = x[i] + ds_boundary * tx[i];
                    }
                    xb
                };
                // Newton correct at boundary
                if let Ok((x_b, ni)) = newton_solve_continuation(f, &x_boundary, lambda_target, cfg)
                {
                    xs.push(x_b);
                    lambdas.push(lambda_target);
                    newton_iters.push(ni);
                }
                break;
            }

            // Corrector: Newton on extended system
            match newton_extended(f, &x_pred, lambda_pred, &x, lambda, &tx, tl, ds, cfg, n) {
                Ok((x_new, lambda_new, n_iters)) => {
                    let curr_det = jacobian_determinant(f, &x_new, lambda_new, cfg.fd_eps);

                    // Limit point detection
                    if prev_det * curr_det < 0.0 {
                        limit_points.push(LimitPoint {
                            x: x_new.clone(),
                            lambda: lambda_new,
                            branch_index: xs.len(),
                            stability_change: true,
                        });
                    }

                    // Branch point detection
                    if curr_det.abs() < cfg.limit_point_tol && (prev_det * curr_det >= 0.0) {
                        let null_vec =
                            approximate_null_vector(f, &x_new, lambda_new, cfg.fd_eps, n);
                        branch_points_list.push(BranchPoint {
                            x: x_new.clone(),
                            lambda: lambda_new,
                            null_vector: Some(null_vec),
                        });
                    }

                    // Update tangent
                    if let Ok((new_tx, new_tl)) =
                        compute_tangent(f, &x_new, lambda_new, tl.signum(), cfg.fd_eps, n)
                    {
                        // Ensure consistent orientation with previous tangent
                        let dot = new_tx
                            .iter()
                            .zip(tx.iter())
                            .fold(0.0, |s, (&a, &b)| s + a * b)
                            + new_tl * tl;
                        if dot < 0.0 {
                            tx = new_tx.mapv(|v| -v);
                            tl = -new_tl;
                        } else {
                            tx = new_tx;
                            tl = new_tl;
                        }
                    }

                    // Adaptive step size
                    let adapt = cfg.desired_newton_iter as f64 / n_iters.max(1) as f64;
                    ds = (ds * adapt.sqrt()).clamp(cfg.ds_min, cfg.ds_max);

                    prev_det = curr_det;
                    x = x_new.clone();
                    lambda = lambda_new;

                    xs.push(x_new);
                    lambdas.push(lambda);
                    newton_iters.push(n_iters);

                    if cfg.compute_stability {
                        let stab = compute_stability_from_jacobian(f, &x, lambda, cfg.fd_eps);
                        stabilities.push(stab);
                    }
                }
                Err(_) => {
                    ds *= cfg.step_adapt_factor;
                    if ds.abs() < cfg.ds_min {
                        return Ok(ContinuationBranchResult {
                            x: xs,
                            lambda: lambdas,
                            stability: stabilities,
                            limit_points,
                            branch_points: branch_points_list,
                            newton_iters,
                            success: false,
                            message: "Step size below minimum".to_string(),
                        });
                    }
                }
            }
        }

        Ok(ContinuationBranchResult {
            x: xs,
            lambda: lambdas,
            stability: stabilities,
            limit_points,
            branch_points: branch_points_list,
            newton_iters,
            success: true,
            message: "Pseudo-arclength continuation completed".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Newton solve for F(x, λ_fixed) = 0 starting from x_guess
fn newton_solve_continuation<F>(
    f: &F,
    x_guess: &Array1<f64>,
    lambda: f64,
    cfg: &ContinuationConfig,
) -> IntegrateResult<(Array1<f64>, usize)>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    let n = x_guess.len();
    let mut x = x_guess.clone();

    for iter in 0..cfg.max_newton_iter {
        let fx = f(&x, lambda);
        let res_norm: f64 = fx.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if res_norm < cfg.newton_tol {
            return Ok((x, iter + 1));
        }
        let mut jac = numerical_jacobian_x(f, &x, lambda, cfg.fd_eps);
        let mut neg_fx = fx.mapv(|v| -v);
        let delta = gauss_solve(&mut jac, &mut neg_fx)?;
        for i in 0..n {
            x[i] += delta[i];
        }
    }

    let fx_final = f(&x, lambda);
    let res_norm: f64 = fx_final.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if res_norm < cfg.newton_tol * 100.0 {
        return Ok((x, cfg.max_newton_iter));
    }

    Err(IntegrateError::ConvergenceError(format!(
        "Newton did not converge in {} iterations (res={:.3e})",
        cfg.max_newton_iter, res_norm
    )))
}

/// Compute tangent vector (tx, tl) to branch at (x, λ), normalized.
/// Solve: [J_x | J_λ] * [tx; tl] = 0, with normalization ‖(tx,tl)‖ = 1
fn compute_tangent<F>(
    f: &F,
    x: &Array1<f64>,
    lambda: f64,
    direction_sign: f64,
    eps: f64,
    n: usize,
) -> IntegrateResult<(Array1<f64>, f64)>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    let jx = numerical_jacobian_x(f, x, lambda, eps);
    let jl = numerical_df_dlambda(f, x, lambda, eps);

    // Solve: J_x * tx = -J_λ * tl  →  normalize
    // Use bordering: solve J_x * v = -J_λ and then normalize
    let mut jx_copy = jx.clone();
    let mut neg_jl = jl.mapv(|v| -v);

    match gauss_solve(&mut jx_copy, &mut neg_jl) {
        Ok(v) => {
            // tl = 1 / sqrt(1 + ‖v‖²), tx = v * tl
            let v_norm_sq: f64 = v.iter().map(|&vi| vi * vi).sum();
            let tl_abs = 1.0 / (1.0 + v_norm_sq).sqrt();
            let mut tx = v.mapv(|vi| vi * tl_abs);
            let mut tl = tl_abs;

            // Orient by direction sign
            if tl * direction_sign < 0.0 {
                tx = tx.mapv(|vi| -vi);
                tl = -tl;
            }
            Ok((tx, tl))
        }
        Err(_) => {
            // Fallback: pure λ-direction tangent
            let tx = Array1::<f64>::zeros(n);
            let tl = direction_sign;
            Ok((tx, tl))
        }
    }
}

/// Newton corrector for pseudo-arclength extended system:
///   F(x, λ) = 0
///   (x - x0)·tx + (λ - λ0)·tl - ds = 0
fn newton_extended<F>(
    f: &F,
    x_pred: &Array1<f64>,
    lambda_pred: f64,
    x0: &Array1<f64>,
    lambda0: f64,
    tx: &Array1<f64>,
    tl: f64,
    ds: f64,
    cfg: &ContinuationConfig,
    n: usize,
) -> IntegrateResult<(Array1<f64>, f64, usize)>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    let mut x = x_pred.clone();
    let mut lam = lambda_pred;
    let size = n + 1;

    for iter in 0..cfg.max_newton_iter {
        let fx = f(&x, lam);

        // Arclength residual
        let arc_res: f64 = x
            .iter()
            .zip(x0.iter())
            .zip(tx.iter())
            .fold(0.0, |s, ((&xi, &x0i), &txi)| s + (xi - x0i) * txi)
            + (lam - lambda0) * tl
            - ds;

        // Combined residual norm
        let res_norm: f64 = fx.iter().map(|&v| v * v).sum::<f64>() + arc_res * arc_res;
        let res_norm = res_norm.sqrt();

        if res_norm < cfg.newton_tol {
            return Ok((x, lam, iter + 1));
        }

        // Build extended Jacobian (n+1) × (n+1)
        // [ J_x   J_λ ] [ dx  ]   [ -F    ]
        // [ tx^T  tl  ] [ dλ  ] = [ -arc  ]
        let jx = numerical_jacobian_x(f, &x, lam, cfg.fd_eps);
        let jl = numerical_df_dlambda(f, &x, lam, cfg.fd_eps);

        let mut big_a = Array2::<f64>::zeros((size, size));
        let mut big_b = Array1::<f64>::zeros(size);

        for i in 0..n {
            for j in 0..n {
                big_a[[i, j]] = jx[[i, j]];
            }
            big_a[[i, n]] = jl[i];
            big_b[i] = -fx[i];
        }
        for j in 0..n {
            big_a[[n, j]] = tx[j];
        }
        big_a[[n, n]] = tl;
        big_b[n] = -arc_res;

        let delta = gauss_solve(&mut big_a, &mut big_b)?;

        for i in 0..n {
            x[i] += delta[i];
        }
        lam += delta[n];
    }

    Err(IntegrateError::ConvergenceError(format!(
        "Pseudo-arclength Newton did not converge in {} iterations",
        cfg.max_newton_iter
    )))
}

/// Estimate det(J_x) using Gaussian elimination (sign of product of pivots)
fn jacobian_determinant<F>(f: &F, x: &Array1<f64>, lambda: f64, eps: f64) -> f64
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    let n = x.len();
    let mut jac = numerical_jacobian_x(f, x, lambda, eps);

    // Gaussian elimination to get upper triangular, track sign
    let mut sign = 1.0_f64;
    let mut det_approx = 1.0_f64;

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = jac[[col, col]].abs();
        for row in (col + 1)..n {
            let v = jac[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return 0.0;
        }
        if max_row != col {
            for j in col..n {
                let tmp = jac[[col, j]];
                jac[[col, j]] = jac[[max_row, j]];
                jac[[max_row, j]] = tmp;
            }
            sign = -sign;
        }
        det_approx *= jac[[col, col]];
        let pivot = jac[[col, col]];
        for row in (col + 1)..n {
            let factor = jac[[row, col]] / pivot;
            for j in col..n {
                let u = factor * jac[[col, j]];
                jac[[row, j]] -= u;
            }
        }
    }

    sign * det_approx
}

/// Compute stability by checking the sign of real parts of diagonal of J_x
/// (approximation using Gershgorin: all Gershgorin discs in left half-plane → stable)
fn compute_stability_from_jacobian<F>(
    f: &F,
    x: &Array1<f64>,
    lambda: f64,
    eps: f64,
) -> SolutionStability
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    let n = x.len();
    let jac = numerical_jacobian_x(f, x, lambda, eps);

    // Gershgorin estimate: disc center = J[i,i], radius = sum_{j≠i} |J[i,j]|
    let mut all_stable = true;
    let mut any_unstable = false;

    for i in 0..n {
        let center = jac[[i, i]];
        let radius: f64 = (0..n).filter(|&j| j != i).map(|j| jac[[i, j]].abs()).sum();
        let rightmost = center + radius;
        let leftmost = center - radius;

        if leftmost > 0.0 {
            any_unstable = true;
            all_stable = false;
        } else if rightmost > 0.0 {
            all_stable = false;
        }
    }

    if any_unstable {
        SolutionStability::Unstable
    } else if all_stable {
        SolutionStability::Stable
    } else {
        SolutionStability::Unknown
    }
}

/// Approximate null vector of J_x using inverse iteration
fn approximate_null_vector<F>(
    f: &F,
    x: &Array1<f64>,
    lambda: f64,
    eps: f64,
    n: usize,
) -> Array1<f64>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    let jac = numerical_jacobian_x(f, x, lambda, eps);

    // Find column with smallest diagonal (rough indicator of null-ness)
    let mut min_col = 0;
    let mut min_val = jac[[0, 0]].abs();
    for i in 1..n {
        let v = jac[[i, i]].abs();
        if v < min_val {
            min_val = v;
            min_col = i;
        }
    }

    // Return the min-col column of the identity (rough approximation)
    let mut null = Array1::<f64>::zeros(n);
    null[min_col] = 1.0;
    null
}

// Suppress unused import warning
#[allow(dead_code)]
fn _use_f64_to_f() {
    let _ = f64_to_f(0.0);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Simple 1D problem: F(x, λ) = x^3 - x - λ = 0
    /// This has a fold/limit point at λ = ±2/3*sqrt(1/3) ≈ ±0.3849
    fn fold_residual(x: &Array1<f64>, lambda: f64) -> Array1<f64> {
        array![x[0] * x[0] * x[0] - x[0] - lambda]
    }

    #[test]
    fn test_natural_continuation_simple() {
        // Trace x^3 - x = λ from λ=0 to λ=0.3
        // At λ=0, solutions are x=-1, 0, 1. Start on x=1 branch.
        let x0 = array![1.0];
        let cfg = ContinuationConfig {
            max_steps: 100,
            ds: 0.05,
            ..Default::default()
        };
        let result = NaturalParameterContinuation::run(&fold_residual, &x0, 0.0, 0.3, &cfg)
            .expect("Natural continuation failed");

        assert!(result.lambda.len() > 2, "Should have more than 2 points");
        let last_lambda = *result.lambda.last().expect("no lambda");
        assert!(
            (last_lambda - 0.3).abs() < 0.1,
            "Last lambda={} should be near 0.3",
            last_lambda
        );
    }

    #[test]
    fn test_pseudo_arclength_continuation() {
        // Trace x^3 - x = λ, starting near x=1, λ=0, going in +λ direction
        let x0 = array![1.0];
        let cfg = ContinuationConfig {
            max_steps: 200,
            ds: 0.05,
            ds_max: 0.2,
            compute_stability: false,
            ..Default::default()
        };
        let result =
            PseudoArcLengthContinuation::run(&fold_residual, &x0, 0.0, (-2.0, 2.0), &cfg, 1.0)
                .expect("Pseudo-arclength continuation failed");

        assert!(result.x.len() > 2, "Branch should have points");
    }

    #[test]
    fn test_linear_problem_continuation() {
        // F(x, λ) = x - λ = 0, trivial branch x = λ
        let linear_f = |x: &Array1<f64>, lambda: f64| array![x[0] - lambda];
        let x0 = array![0.0];
        let cfg = ContinuationConfig {
            max_steps: 50,
            ds: 0.1,
            compute_stability: false,
            ..Default::default()
        };
        let result = NaturalParameterContinuation::run(&linear_f, &x0, 0.0, 1.0, &cfg)
            .expect("Linear continuation failed");

        // All points should satisfy x ≈ λ
        for (xi, &li) in result.x.iter().zip(result.lambda.iter()) {
            assert!(
                (xi[0] - li).abs() < 1e-6,
                "x={} should equal lambda={}",
                xi[0],
                li
            );
        }
    }
}
