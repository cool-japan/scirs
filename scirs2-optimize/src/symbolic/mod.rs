//! Symbolic-gradient-based optimization.
//!
//! Use `scirs2-symbolic`'s symbolic gradient + Hessian as the input to
//! Newton's method, L-BFGS, and trust-region solvers — avoiding finite-difference
//! approximations and their associated step-size sensitivity.
//!
//! # Example
//! ```no_run
//! use scirs2_optimize::symbolic::newton;
//! use scirs2_symbolic::eml::LoweredOp;
//!
//! // Cost function: f(x) = x² (minimum at x=0)
//! let cost = LoweredOp::Mul(
//!     Box::new(LoweredOp::Var(0)),
//!     Box::new(LoweredOp::Var(0)),
//! );
//! let result = newton(&cost, &[5.0], 50, 1e-8).expect("converge");
//! assert!(result.x[0].abs() < 1e-6);
//! ```

use std::sync::Arc;

use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::{grad, hessian, LoweredOp};

/// Result of a symbolic optimization run.
#[derive(Debug, Clone)]
pub struct SymbolicNewtonResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Final cost function value.
    pub f_final: f64,
    /// Number of iterations performed.
    pub iters: usize,
    /// True if convergence tolerance was reached.
    pub converged: bool,
}

/// Errors from symbolic Newton.
#[derive(Debug, thiserror::Error)]
pub enum SymbolicNewtonError {
    /// Underlying `LoweredOp` evaluation failure (domain, division by zero,
    /// unbound variable, etc.).
    #[error("evaluation error: {0}")]
    EvalError(String),
    /// Hessian matrix is singular (or numerically too ill-conditioned) so the
    /// Newton system H·dx = -grad cannot be solved.
    #[error("singular Hessian — cannot invert")]
    SingularHessian,
    /// Initial point dimension does not match the cost function's variable count.
    #[error("dimension mismatch: x0 has {x0_len}, cost expects {n_vars}")]
    DimMismatch {
        /// Length of the supplied `x0` vector.
        x0_len: usize,
        /// Number of variables `cost.count_vars()` reported.
        n_vars: usize,
    },
}

/// Newton's method using scirs2-symbolic gradient and Hessian.
///
/// # Arguments
/// - `cost`: scalar cost function as a `LoweredOp` (must be twice-differentiable)
/// - `x0`: initial point, length matches `cost.count_vars()`
/// - `max_iter`: max iterations
/// - `tol`: convergence tolerance on `||grad||₂`
///
/// # Errors
/// - [`SymbolicNewtonError::DimMismatch`] when `x0.len() < cost.count_vars()`
/// - [`SymbolicNewtonError::EvalError`] when symbolic gradient/Hessian
///   evaluation fails (domain violation, unbound variable, etc.)
/// - [`SymbolicNewtonError::SingularHessian`] when the Hessian is singular at
///   the current iterate
pub fn newton(
    cost: &LoweredOp,
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<SymbolicNewtonResult, SymbolicNewtonError> {
    let n_vars = cost.count_vars();
    if x0.len() < n_vars {
        return Err(SymbolicNewtonError::DimMismatch {
            x0_len: x0.len(),
            n_vars,
        });
    }

    // Precompute symbolic grad and Hessian — once
    let grad_ops: Vec<LoweredOp> = (0..n_vars).map(|i| grad(cost, i)).collect();
    let hess_ops: Vec<Vec<LoweredOp>> = hessian(cost, n_vars);

    let mut x = x0.to_vec();
    let mut converged = false;
    let mut iters = 0;

    for k in 0..max_iter {
        iters = k + 1;
        let ctx = EvalCtx::new(&x);

        // Evaluate gradient
        let mut grad_vec: Vec<f64> = Vec::with_capacity(n_vars);
        for g_op in &grad_ops {
            let v =
                eval_real(g_op, &ctx).map_err(|e| SymbolicNewtonError::EvalError(e.to_string()))?;
            grad_vec.push(v);
        }

        // Check convergence: ||grad||₂ < tol
        let grad_norm: f64 = grad_vec.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol {
            converged = true;
            break;
        }

        // Evaluate Hessian
        let mut hess_mat: Vec<Vec<f64>> = Vec::with_capacity(n_vars);
        for row in &hess_ops {
            let mut hess_row: Vec<f64> = Vec::with_capacity(n_vars);
            for h_op in row {
                let v = eval_real(h_op, &ctx)
                    .map_err(|e| SymbolicNewtonError::EvalError(e.to_string()))?;
                hess_row.push(v);
            }
            hess_mat.push(hess_row);
        }

        // Solve H·dx = -grad → solve_linear(H, grad) returns H⁻¹·grad
        let dx = solve_linear(&hess_mat, &grad_vec)?;

        // Update: x ← x - dx (Newton step is dx = H⁻¹·grad, so x ← x - H⁻¹·grad)
        for (xi, dxi) in x.iter_mut().zip(dx.iter()) {
            *xi -= *dxi;
        }
    }

    let final_ctx = EvalCtx::new(&x);
    let f_final =
        eval_real(cost, &final_ctx).map_err(|e| SymbolicNewtonError::EvalError(e.to_string()))?;

    Ok(SymbolicNewtonResult {
        x,
        f_final,
        iters,
        converged,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Shared result / error types for L-BFGS and trust-region symbolic solvers
// ────────────────────────────────────────────────────────────────────────────

/// Result returned by [`lbfgs_symbolic`] and [`trust_region_symbolic`].
#[derive(Debug, Clone)]
pub struct SymbolicOptResult {
    /// Solution vector.
    pub x: scirs2_core::ndarray::Array1<f64>,
    /// Objective value at `x`.
    pub f_val: f64,
    /// L₂ norm of the gradient at `x`.
    pub grad_norm: f64,
    /// Number of iterations performed.
    pub iters: usize,
    /// True when `grad_norm < tol`.
    pub converged: bool,
}

/// Errors from [`lbfgs_symbolic`] and [`trust_region_symbolic`].
#[derive(Debug, thiserror::Error)]
pub enum SymbolicOptError {
    /// Objective evaluation failed.
    #[error("eval error: {0}")]
    EvalError(String),
    /// Symbolic gradient evaluation failed.
    #[error("gradient eval error: {0}")]
    GradEvalError(String),
    /// Symbolic Hessian evaluation failed.
    #[error("Hessian eval error: {0}")]
    HessEvalError(String),
    /// `x0.len()` does not match the number of variables in the objective.
    #[error("dimension mismatch: expected {expected} variables, got {got}")]
    DimMismatch {
        /// Variables inferred from the objective.
        expected: usize,
        /// Length of the supplied `x0`.
        got: usize,
    },
    /// Optimizer exhausted its iteration budget without reaching `tol`.
    #[error("not converged after {iters} iters (grad_norm = {grad_norm:.6e})")]
    NotConverged {
        /// Iteration count at termination.
        iters: usize,
        /// Gradient norm at termination.
        grad_norm: f64,
    },
    /// Wolfe-condition line search failed to find a suitable step.
    #[error("line search failed")]
    LineSearchFailed,
}

// ────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ────────────────────────────────────────────────────────────────────────────

/// Evaluate `grad_ops` at `x`, returning an error variant on failure.
fn eval_gradient(grad_ops: &[LoweredOp], x: &[f64]) -> Result<Vec<f64>, SymbolicOptError> {
    let ctx = EvalCtx::new(x);
    let mut g = Vec::with_capacity(grad_ops.len());
    for op in grad_ops {
        let v = eval_real(op, &ctx).map_err(|e| SymbolicOptError::GradEvalError(e.to_string()))?;
        g.push(v);
    }
    Ok(g)
}

/// Evaluate objective at `x`.
fn eval_objective(obj: &LoweredOp, x: &[f64]) -> Result<f64, SymbolicOptError> {
    let ctx = EvalCtx::new(x);
    eval_real(obj, &ctx).map_err(|e| SymbolicOptError::EvalError(e.to_string()))
}

/// L₂ norm of a slice.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Dot product.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

// ────────────────────────────────────────────────────────────────────────────
// L-BFGS with exact symbolic gradient
// ────────────────────────────────────────────────────────────────────────────

/// L-BFGS optimizer using exact symbolic gradient.
///
/// Converges faster than L-BFGS with finite-difference gradients because the
/// gradient is exact (no step-size tuning needed). Uses a two-loop L-BFGS
/// recursion with Wolfe-condition line search.
///
/// # Arguments
/// - `objective`: scalar objective as an `Arc<LoweredOp>`
/// - `x0`: initial point; length must equal `objective.count_vars()`
/// - `max_iter`: maximum number of iterations
/// - `tol`: convergence tolerance on `||∇f||₂`
/// - `memory`: L-BFGS history length (5–20 is typical)
///
/// # Errors
/// - [`SymbolicOptError::DimMismatch`] when `x0.len() != objective.count_vars()`
/// - [`SymbolicOptError::EvalError`] / [`SymbolicOptError::GradEvalError`] on
///   symbolic evaluation failures
/// - [`SymbolicOptError::NotConverged`] when `max_iter` is exhausted
///
/// # Example
/// ```no_run
/// # #[cfg(feature = "symbolic")]
/// # {
/// use std::sync::Arc;
/// use scirs2_optimize::symbolic::lbfgs_symbolic;
/// use scirs2_symbolic::eml::LoweredOp;
/// use scirs2_core::ndarray::array;
///
/// let obj = Arc::new(LoweredOp::Mul(
///     Box::new(LoweredOp::Var(0)),
///     Box::new(LoweredOp::Var(0)),
/// ));
/// let result = lbfgs_symbolic(&obj, array![3.0].view(), 100, 1e-8, 10).expect("converge");
/// assert!(result.x[0].abs() < 1e-6);
/// # }
/// ```
pub fn lbfgs_symbolic(
    objective: &Arc<LoweredOp>,
    x0: scirs2_core::ndarray::ArrayView1<f64>,
    max_iter: usize,
    tol: f64,
    memory: usize,
) -> Result<SymbolicOptResult, SymbolicOptError> {
    use scirs2_core::ndarray::Array1;

    let n_vars = objective.count_vars();
    if x0.len() != n_vars {
        return Err(SymbolicOptError::DimMismatch {
            expected: n_vars,
            got: x0.len(),
        });
    }

    // Precompute symbolic gradient ops once.
    let grad_ops: Vec<LoweredOp> = (0..n_vars).map(|i| grad(objective.as_ref(), i)).collect();

    let mut x: Vec<f64> = x0.iter().copied().collect();

    // Evaluate initial gradient before the loop.
    let mut g = eval_gradient(&grad_ops, &x)?;
    let mut grad_norm = l2_norm(&g);

    // Handle max_iter == 0 immediately.
    if max_iter == 0 {
        return Err(SymbolicOptError::NotConverged {
            iters: 0,
            grad_norm,
        });
    }

    if grad_norm < tol {
        let f_val = eval_objective(objective.as_ref(), &x)?;
        return Ok(SymbolicOptResult {
            x: Array1::from_vec(x),
            f_val,
            grad_norm,
            iters: 0,
            converged: true,
        });
    }

    // History: ring buffer of (s_k, y_k, rho_k) triples.
    // s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k, rho_k = 1 / (y_k^T s_k)
    let mem = memory.max(1);
    let mut history: std::collections::VecDeque<(Vec<f64>, Vec<f64>, f64)> =
        std::collections::VecDeque::with_capacity(mem);

    let mut converged = false;
    let mut iters = 0;

    for k in 0..max_iter {
        iters = k + 1;

        // ── Two-loop L-BFGS recursion ────────────────────────────────────
        let direction = lbfgs_direction(&g, &history);

        // ── Wolfe-condition line search ───────────────────────────────────
        let f_k = eval_objective(objective.as_ref(), &x)?;
        let dg = dot(&g, &direction); // ∇f^T d (should be < 0)

        const C1: f64 = 1e-4; // sufficient decrease
        const C2: f64 = 0.9; // curvature
        const MAX_LS: usize = 20;

        let mut alpha = 1.0_f64;
        let mut x_new: Vec<f64> = Vec::with_capacity(n_vars);
        let mut g_new: Vec<f64>;
        let mut armijo_ok = false;

        // Try to satisfy both Wolfe conditions with backtracking.
        let mut ls_iter = 0;
        loop {
            x_new = x
                .iter()
                .zip(&direction)
                .map(|(xi, di)| xi + alpha * di)
                .collect();
            let f_new = eval_objective(objective.as_ref(), &x_new)?;
            let suff_dec = f_new <= f_k + C1 * alpha * dg;

            if suff_dec {
                g_new = eval_gradient(&grad_ops, &x_new)?;
                let curvature = dot(&g_new, &direction).abs() <= C2 * dg.abs();
                if curvature {
                    armijo_ok = true;
                    break;
                }
                // Armijo satisfied but curvature not — still usable.
                armijo_ok = true;
                // Continue backtracking hoping to also satisfy curvature.
            } else {
                g_new = Vec::new(); // placeholder — will recompute on acceptance
            }

            ls_iter += 1;
            if ls_iter >= MAX_LS {
                break;
            }
            alpha *= 0.5;
        }

        // If we left the loop without a good step, accept Armijo-only or fail.
        if !armijo_ok {
            // Accept the last alpha that passed Armijo (may be very small).
            x_new = x
                .iter()
                .zip(&direction)
                .map(|(xi, di)| xi + alpha * di)
                .collect();
            let f_new = eval_objective(objective.as_ref(), &x_new)?;
            if f_new > f_k + C1 * alpha * dg {
                return Err(SymbolicOptError::LineSearchFailed);
            }
            g_new = eval_gradient(&grad_ops, &x_new)?;
        } else if g_new.is_empty() {
            // Filled placeholder path — curvature branch that broke early
            g_new = eval_gradient(&grad_ops, &x_new)?;
        }

        // ── Update history ────────────────────────────────────────────────
        let s_k: Vec<f64> = x_new.iter().zip(&x).map(|(xn, xo)| xn - xo).collect();
        let y_k: Vec<f64> = g_new.iter().zip(&g).map(|(gn, go)| gn - go).collect();
        let sy = dot(&s_k, &y_k);
        if sy.abs() > 1e-20 {
            let rho = 1.0 / sy;
            if history.len() == mem {
                history.pop_front();
            }
            history.push_back((s_k, y_k, rho));
        }

        // ── Advance ───────────────────────────────────────────────────────
        x = x_new;
        g = g_new;
        grad_norm = l2_norm(&g);

        if grad_norm < tol {
            converged = true;
            break;
        }
    }

    let f_val = eval_objective(objective.as_ref(), &x)?;

    if converged {
        Ok(SymbolicOptResult {
            x: Array1::from_vec(x),
            f_val,
            grad_norm,
            iters,
            converged: true,
        })
    } else {
        Err(SymbolicOptError::NotConverged { iters, grad_norm })
    }
}

/// Compute the L-BFGS search direction given the current gradient and history.
///
/// Standard two-loop recursion; returns `-H_k · g`.
fn lbfgs_direction(
    g: &[f64],
    history: &std::collections::VecDeque<(Vec<f64>, Vec<f64>, f64)>,
) -> Vec<f64> {
    let n = g.len();
    let m = history.len();
    let mut q: Vec<f64> = g.to_vec();
    let mut alphas: Vec<f64> = vec![0.0; m];

    // First loop (newest to oldest)
    for (idx, (s, y, rho)) in history.iter().rev().enumerate() {
        let a = rho * dot(s, &q);
        alphas[m - 1 - idx] = a;
        for j in 0..n {
            q[j] -= a * y[j];
        }
    }

    // Initial Hessian scaling γ_k = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1}
    let mut r: Vec<f64> = if let Some((s, y, _)) = history.back() {
        let sy = dot(s, y);
        let yy = dot(y, y);
        let gamma = if yy > 1e-20 { sy / yy } else { 1.0 };
        q.iter().map(|qi| gamma * qi).collect()
    } else {
        q.clone() // H_0 = I
    };

    // Second loop (oldest to newest)
    for (idx, (s, y, rho)) in history.iter().enumerate() {
        let beta = rho * dot(y, &r);
        let a = alphas[idx];
        for j in 0..n {
            r[j] += s[j] * (a - beta);
        }
    }

    // Search direction = -r
    r.iter().map(|ri| -ri).collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Trust-region dogleg with exact symbolic gradient + Hessian
// ────────────────────────────────────────────────────────────────────────────

/// Trust-region optimizer using exact symbolic gradient + Hessian.
///
/// Uses the dogleg step: if the Newton step is within the trust radius, take it;
/// otherwise interpolate between the steepest-descent step and the Newton step
/// on the trust-region boundary.
///
/// # Arguments
/// - `objective`: scalar objective as an `Arc<LoweredOp>`
/// - `x0`: initial point; length must equal `objective.count_vars()`
/// - `max_iter`: maximum number of iterations
/// - `tol`: convergence tolerance on `||∇f||₂`
/// - `initial_radius`: starting trust-region radius (typically 1.0)
///
/// # Errors
/// - [`SymbolicOptError::DimMismatch`] when `x0.len() != objective.count_vars()`
/// - [`SymbolicOptError::EvalError`] / [`SymbolicOptError::GradEvalError`] /
///   [`SymbolicOptError::HessEvalError`] on symbolic evaluation failures
/// - [`SymbolicOptError::NotConverged`] when `max_iter` is exhausted
///
/// # Example
/// ```no_run
/// # #[cfg(feature = "symbolic")]
/// # {
/// use std::sync::Arc;
/// use scirs2_optimize::symbolic::trust_region_symbolic;
/// use scirs2_symbolic::eml::LoweredOp;
/// use scirs2_core::ndarray::array;
///
/// let obj = Arc::new(LoweredOp::Mul(
///     Box::new(LoweredOp::Var(0)),
///     Box::new(LoweredOp::Var(0)),
/// ));
/// let result = trust_region_symbolic(&obj, array![3.0].view(), 100, 1e-8, 1.0)
///     .expect("converge");
/// assert!(result.x[0].abs() < 1e-6);
/// # }
/// ```
pub fn trust_region_symbolic(
    objective: &Arc<LoweredOp>,
    x0: scirs2_core::ndarray::ArrayView1<f64>,
    max_iter: usize,
    tol: f64,
    initial_radius: f64,
) -> Result<SymbolicOptResult, SymbolicOptError> {
    use scirs2_core::ndarray::Array1;

    let n_vars = objective.count_vars();
    if x0.len() != n_vars {
        return Err(SymbolicOptError::DimMismatch {
            expected: n_vars,
            got: x0.len(),
        });
    }

    // Precompute symbolic gradient + Hessian ops once.
    let grad_ops: Vec<LoweredOp> = (0..n_vars).map(|i| grad(objective.as_ref(), i)).collect();
    let hess_ops: Vec<Vec<LoweredOp>> = hessian(objective.as_ref(), n_vars);

    let mut x: Vec<f64> = x0.iter().copied().collect();

    // Evaluate initial gradient.
    let mut g = eval_gradient(&grad_ops, &x)?;
    let mut grad_norm = l2_norm(&g);

    if max_iter == 0 {
        return Err(SymbolicOptError::NotConverged {
            iters: 0,
            grad_norm,
        });
    }

    if grad_norm < tol {
        let f_val = eval_objective(objective.as_ref(), &x)?;
        return Ok(SymbolicOptResult {
            x: Array1::from_vec(x),
            f_val,
            grad_norm,
            iters: 0,
            converged: true,
        });
    }

    let mut delta = initial_radius.max(1e-8);
    let mut converged = false;
    let mut iters = 0;

    for k in 0..max_iter {
        iters = k + 1;

        let f_k = eval_objective(objective.as_ref(), &x)?;

        // Evaluate Hessian at current x.
        let ctx = EvalCtx::new(&x);
        let mut h_mat: Vec<Vec<f64>> = Vec::with_capacity(n_vars);
        for row in &hess_ops {
            let mut h_row: Vec<f64> = Vec::with_capacity(n_vars);
            for h_op in row {
                let v = eval_real(h_op, &ctx)
                    .map_err(|e| SymbolicOptError::HessEvalError(e.to_string()))?;
                h_row.push(v);
            }
            h_mat.push(h_row);
        }

        // Compute dogleg step.
        let p = dogleg_step_sym(&g, &h_mat, delta);

        // Actual vs predicted reduction.
        let x_new: Vec<f64> = x.iter().zip(&p).map(|(xi, pi)| xi + pi).collect();
        let f_new = eval_objective(objective.as_ref(), &x_new)?;
        let actual_red = f_k - f_new;

        // Predicted reduction: m(0) - m(p) = -g^T p - 0.5 p^T H p
        let gtp = dot(&g, &p);
        let htp: Vec<f64> = matvec(&h_mat, &p);
        let phtp = dot(&p, &htp);
        let pred_red = -(gtp + 0.5 * phtp);

        // Update trust radius.
        let rho = if pred_red.abs() < 1e-20 {
            1.0
        } else {
            actual_red / pred_red
        };

        if rho >= 0.1 {
            // Accept step.
            x = x_new;
            g = eval_gradient(&grad_ops, &x)?;
            grad_norm = l2_norm(&g);

            if grad_norm < tol {
                converged = true;
                // Expand radius and break.
                if rho >= 0.75 {
                    let p_norm = l2_norm(&p);
                    if p_norm >= 0.8 * delta {
                        delta = (2.0 * delta).min(100.0);
                    }
                }
                break;
            }
        }
        // Else: reject step, x unchanged, g unchanged.

        // Adjust radius.
        if rho >= 0.75 {
            let p_norm = l2_norm(&p);
            if p_norm >= 0.8 * delta {
                delta = (2.0 * delta).min(100.0);
            }
        } else if rho < 0.25 {
            delta *= 0.25;
        }

        if delta < 1e-14 {
            // Trust region collapsed — treat as not converged.
            break;
        }
    }

    let f_val = eval_objective(objective.as_ref(), &x)?;

    if converged {
        Ok(SymbolicOptResult {
            x: Array1::from_vec(x),
            f_val,
            grad_norm,
            iters,
            converged: true,
        })
    } else {
        Err(SymbolicOptError::NotConverged { iters, grad_norm })
    }
}

/// Compute the dogleg step p for the trust-region subproblem:
///   min_p  g^T p + 0.5 p^T H p   s.t.  ||p|| ≤ Δ
///
/// Falls back to the steepest-descent Cauchy step when the Hessian is
/// indefinite or the Newton system is singular.
fn dogleg_step_sym(g: &[f64], h: &[Vec<f64>], delta: f64) -> Vec<f64> {
    let n = g.len();

    // Newton step: p_N = -H⁻¹ g  (solve H p_N = -g via Gaussian elimination)
    let neg_g: Vec<f64> = g.iter().map(|gi| -gi).collect();
    let p_n_opt = solve_linear_opt(h, &neg_g);

    // Steepest-descent step: p_SD = -(g^T g / g^T H g) * g
    let gtg = dot(g, g);
    let hg = matvec(h, g);
    let gthg = dot(g, &hg); // g^T H g

    let p_sd: Vec<f64> = if gthg > 1e-20 {
        let scale = gtg / gthg;
        g.iter().map(|gi| -scale * gi).collect()
    } else {
        // H is indefinite or zero — use steepest-descent direction with unit step.
        let gn = gtg.sqrt().max(1e-20);
        g.iter().map(|gi| -gi / gn).collect()
    };

    let p_sd_norm = l2_norm(&p_sd);

    // If Newton system succeeded, attempt full dogleg.
    if let Some(p_n) = p_n_opt {
        let p_n_norm = l2_norm(&p_n);

        if p_n_norm <= delta {
            // Newton step fits — take it.
            return p_n;
        }

        if p_sd_norm >= delta {
            // Cauchy point already outside trust region — scale SD.
            return p_sd.iter().map(|pi| delta / p_sd_norm * pi).collect();
        }

        // Interpolate: p = p_SD + τ * (p_N - p_SD), ||p|| = Δ
        // ||p_SD + τ * d||² = Δ²  where d = p_N - p_SD
        // ||p_SD||² + 2τ p_SD^T d + τ² ||d||² = Δ²
        let d: Vec<f64> = p_n.iter().zip(&p_sd).map(|(pni, psi)| pni - psi).collect();
        let dd = dot(&d, &d);
        let psd_d = dot(&p_sd, &d);
        let psd2 = p_sd_norm * p_sd_norm;
        let discriminant = psd_d * psd_d - dd * (psd2 - delta * delta);
        let tau = if dd < 1e-20 || discriminant < 0.0 {
            1.0
        } else {
            (-psd_d + discriminant.sqrt()) / dd
        };
        let tau = tau.clamp(0.0, 1.0);
        p_sd.iter()
            .zip(&d)
            .map(|(psi, di)| psi + tau * di)
            .collect()
    } else {
        // Singular Hessian — Cauchy step.
        if p_sd_norm <= delta {
            p_sd
        } else {
            p_sd.iter().map(|pi| delta / p_sd_norm * pi).collect()
        }
    }
}

/// Matrix-vector product H · v.
fn matvec(h: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    h.iter().map(|row| dot(row, v)).collect()
}

/// Partial-pivoting Gaussian elimination. Returns `None` when singular.
fn solve_linear_opt(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Some(Vec::new());
    }

    let mut mat: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = mat[k][k].abs();
        for i in (k + 1)..n {
            let v = mat[i][k].abs();
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        if max_val < 1e-12 {
            return None; // singular
        }
        mat.swap(k, max_idx);
        for i in (k + 1)..n {
            let factor = mat[i][k] / mat[k][k];
            for j in k..(n + 1) {
                mat[i][j] -= factor * mat[k][j];
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = mat[i][n];
        for j in (i + 1)..n {
            sum -= mat[i][j] * x[j];
        }
        x[i] = sum / mat[i][i];
    }
    Some(x)
}

/// Solve linear system A·x = b via partial-pivoting Gaussian elimination.
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, SymbolicNewtonError> {
    let n = b.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Augmented matrix
    let mut mat: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = mat[k][k].abs();
        for i in (k + 1)..n {
            let v = mat[i][k].abs();
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        if max_val < 1e-12 {
            return Err(SymbolicNewtonError::SingularHessian);
        }
        // Swap
        mat.swap(k, max_idx);
        // Eliminate
        for i in (k + 1)..n {
            let factor = mat[i][k] / mat[k][k];
            for j in k..(n + 1) {
                mat[i][j] -= factor * mat[k][j];
            }
        }
    }

    // Back-substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = mat[i][n];
        for j in (i + 1)..n {
            sum -= mat[i][j] * x[j];
        }
        x[i] = sum / mat[i][i];
    }

    Ok(x)
}

// ────────────────────────────────────────────────────────────────────────────
// Symbolic Lagrangian and KKT conditions
// ────────────────────────────────────────────────────────────────────────────

/// KKT system for equality-constrained optimization: min f(x) s.t. gᵢ(x) = 0.
///
/// Variables Var(0..n_vars) are the primal variables x.
/// Variables Var(n_vars..n_vars+n_constraints) are the Lagrange multipliers λᵢ.
pub struct KktSystem {
    /// Lagrangian L = f + Σ λᵢ·gᵢ.
    pub lagrangian: LoweredOp,
    /// Stationarity conditions: ∂L/∂xⱼ = 0 for j in 0..n_vars.
    pub stationarity: Vec<LoweredOp>,
    /// Constraint residuals: gᵢ(x) for i in 0..n_constraints.
    pub constraint_residuals: Vec<LoweredOp>,
    /// Number of primal variables.
    pub n_vars: usize,
    /// Number of equality constraints.
    pub n_constraints: usize,
}

/// Errors from [`build_kkt`].
#[derive(Debug)]
pub enum LagrangianError {
    /// Variable or constraint dimension was unexpected.
    DimMismatch {
        /// Expected number of variables.
        expected: usize,
        /// Actual number encountered.
        got: usize,
    },
    /// Error while computing a symbolic gradient.
    GradError(String),
}

/// Form the KKT system for equality-constrained optimization symbolically.
///
/// Builds the Lagrangian L = f + Σ λᵢ·gᵢ and derives stationarity conditions
/// ∂L/∂xⱼ = 0 for each primal variable.
///
/// # Arguments
/// - `objective`: scalar objective f(x)
/// - `constraints`: slice of equality constraints gᵢ(x) (each = 0 at a solution)
/// - `n_vars`: number of primal decision variables
///
/// # Errors
/// Returns [`LagrangianError::DimMismatch`] when the objective or a constraint
/// mentions more variables than `n_vars`.
pub fn build_kkt(
    objective: &Arc<LoweredOp>,
    constraints: &[Arc<LoweredOp>],
    n_vars: usize,
) -> Result<KktSystem, LagrangianError> {
    let m = constraints.len();

    // Validate: no operand should reference a variable index >= n_vars
    // (multiplier slots n_vars..n_vars+m are reserved by the KKT construction).
    let obj_vars = objective.count_vars();
    if obj_vars > n_vars {
        return Err(LagrangianError::DimMismatch {
            expected: n_vars,
            got: obj_vars,
        });
    }

    // Build lagrangian = f + Σ λᵢ·gᵢ
    // λᵢ occupies Var(n_vars + i).
    let mut lagrangian: LoweredOp = objective.as_ref().clone();
    for (i, g) in constraints.iter().enumerate() {
        let lam_i = LoweredOp::Var(n_vars + i);
        let term = LoweredOp::Mul(Box::new(lam_i), Box::new(g.as_ref().clone()));
        lagrangian = LoweredOp::Add(Box::new(lagrangian), Box::new(term));
    }

    // Stationarity: ∂L/∂xⱼ for j in 0..n_vars
    let stationarity: Vec<LoweredOp> = (0..n_vars).map(|j| grad(&lagrangian, j)).collect();

    // Constraint residuals: just the gᵢ themselves
    let constraint_residuals: Vec<LoweredOp> =
        constraints.iter().map(|g| g.as_ref().clone()).collect();

    Ok(KktSystem {
        lagrangian,
        stationarity,
        constraint_residuals,
        n_vars,
        n_constraints: m,
    })
}

/// Solve an equality-constrained optimization via Newton on the KKT system.
///
/// Forms the Lagrangian symbolically, derives KKT stationarity conditions, then
/// applies Newton's method on the full nonlinear KKT system
/// `[∂L/∂x; g(x)] = 0`.
///
/// # Arguments
/// - `objective`: scalar objective f(x)
/// - `constraints`: equality constraints gᵢ(x) = 0
/// - `x0`: initial primal point; length must equal `count_vars(objective)`
/// - `lambda0`: initial multiplier guess; length must equal `constraints.len()`
/// - `max_iter`: maximum Newton iterations
/// - `tol`: convergence tolerance on the L₂ norm of the KKT residual
///
/// # Errors
/// - [`SymbolicOptError::DimMismatch`] when `x0.len()` or `lambda0.len()` is wrong
/// - [`SymbolicOptError::EvalError`] on symbolic evaluation failures
/// - [`SymbolicOptError::NotConverged`] when `max_iter` is exhausted
pub fn solve_lagrangian_symbolic(
    objective: &Arc<LoweredOp>,
    constraints: &[Arc<LoweredOp>],
    x0: scirs2_core::ndarray::ArrayView1<f64>,
    lambda0: scirs2_core::ndarray::ArrayView1<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<SymbolicOptResult, SymbolicOptError> {
    use scirs2_core::ndarray::Array1;

    let m = constraints.len();

    // Infer n_vars as the maximum variable index referenced across objective
    // and all constraints.
    let n_vars = std::iter::once(objective.count_vars())
        .chain(constraints.iter().map(|c| c.count_vars()))
        .max()
        .unwrap_or(0);

    if x0.len() != n_vars {
        return Err(SymbolicOptError::DimMismatch {
            expected: n_vars,
            got: x0.len(),
        });
    }
    if lambda0.len() != m {
        return Err(SymbolicOptError::DimMismatch {
            expected: m,
            got: lambda0.len(),
        });
    }

    // Build KKT system — map error variants
    let kkt = build_kkt(objective, constraints, n_vars).map_err(|e| match e {
        LagrangianError::DimMismatch { expected, got } => {
            SymbolicOptError::DimMismatch { expected, got }
        }
        LagrangianError::GradError(s) => SymbolicOptError::GradEvalError(s),
    })?;

    // Total unknowns: N = n_vars + m
    let big_n = n_vars + m;

    // Combined variable vector z = [x0..., lambda0...]
    let mut z: Vec<f64> = x0.iter().copied().chain(lambda0.iter().copied()).collect();

    // All KKT equations: stationarity (n_vars eqs) ++ constraint residuals (m eqs)
    let eqs: Vec<&LoweredOp> = kkt
        .stationarity
        .iter()
        .chain(kkt.constraint_residuals.iter())
        .collect();

    // Precompute symbolic Jacobian J[i][j] = ∂eqs[i]/∂z[j] — once before the loop.
    let jac_ops: Vec<Vec<LoweredOp>> = eqs
        .iter()
        .map(|eq| (0..big_n).map(|j| grad(eq, j)).collect())
        .collect();

    // Evaluate initial residual for max_iter==0 early exit.
    let initial_residual = {
        let ctx = EvalCtx::new(&z);
        let mut f_vec = Vec::with_capacity(big_n);
        for eq in &eqs {
            let v = eval_real(eq, &ctx).map_err(|e| SymbolicOptError::EvalError(e.to_string()))?;
            f_vec.push(v);
        }
        f_vec
    };
    let mut residual_norm = l2_norm(&initial_residual);

    if max_iter == 0 {
        return Err(SymbolicOptError::NotConverged {
            iters: 0,
            grad_norm: residual_norm,
        });
    }

    // Check if already converged at the initial point.
    if residual_norm < tol {
        let f_val = eval_objective(objective.as_ref(), &z[..n_vars])?;
        return Ok(SymbolicOptResult {
            x: Array1::from_vec(z[..n_vars].to_vec()),
            f_val,
            grad_norm: residual_norm,
            iters: 0,
            converged: true,
        });
    }

    let mut converged = false;
    let mut iters = 0;

    for k in 0..max_iter {
        iters = k + 1;

        // Evaluate KKT residual vector F
        let ctx = EvalCtx::new(&z);
        let mut f_vec: Vec<f64> = Vec::with_capacity(big_n);
        for eq in &eqs {
            let v = eval_real(eq, &ctx).map_err(|e| SymbolicOptError::EvalError(e.to_string()))?;
            f_vec.push(v);
        }

        residual_norm = l2_norm(&f_vec);
        if residual_norm < tol {
            converged = true;
            break;
        }

        // Evaluate Jacobian matrix J
        let mut jac_mat: Vec<Vec<f64>> = Vec::with_capacity(big_n);
        for row_ops in &jac_ops {
            let mut row: Vec<f64> = Vec::with_capacity(big_n);
            for op in row_ops {
                let v =
                    eval_real(op, &ctx).map_err(|e| SymbolicOptError::EvalError(e.to_string()))?;
                row.push(v);
            }
            jac_mat.push(row);
        }

        // Solve J · dz = -F
        let neg_f: Vec<f64> = f_vec.iter().map(|v| -v).collect();
        let dz = match solve_linear_opt(&jac_mat, &neg_f) {
            Some(d) => d,
            None => {
                // Singular Jacobian — fall back to negative residual step (gradient descent).
                neg_f
            }
        };

        // Update z ← z + dz
        for (zi, dzi) in z.iter_mut().zip(dz.iter()) {
            *zi += dzi;
        }
    }

    let f_val = eval_objective(objective.as_ref(), &z[..n_vars])?;

    if converged {
        Ok(SymbolicOptResult {
            x: Array1::from_vec(z[..n_vars].to_vec()),
            f_val,
            grad_norm: residual_norm,
            iters,
            converged: true,
        })
    } else {
        Err(SymbolicOptError::NotConverged {
            iters,
            grad_norm: residual_norm,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Closed-form quadratic line-search wrapper
// ─────────────────────────────────────────────────────────────────────────────

pub mod line_search;
pub use line_search::{OptLineSearchError, SymbolicLineSearch};

#[cfg(test)]
mod tests {
    use super::*;

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    #[test]
    fn newton_x_squared_one_step() {
        // f(x) = x² → grad = 2x, hess = 2; Newton step: dx = 2x / 2 = x; x_new = x - x = 0
        let cost = LoweredOp::Mul(Box::new(var(0)), Box::new(var(0)));
        let result = newton(&cost, &[5.0], 10, 1e-8).expect("converge");
        assert!(result.converged);
        assert!(result.x[0].abs() < 1e-6, "x = {}", result.x[0]);
        assert!(result.iters <= 2);
    }

    #[test]
    fn newton_quadratic_2d() {
        // f(x, y) = x² + y² → minimum at (0, 0)
        let cost = LoweredOp::Add(
            Box::new(LoweredOp::Mul(Box::new(var(0)), Box::new(var(0)))),
            Box::new(LoweredOp::Mul(Box::new(var(1)), Box::new(var(1)))),
        );
        let result = newton(&cost, &[3.0, -4.0], 10, 1e-8).expect("converge");
        assert!(result.converged);
        assert!(result.x[0].abs() < 1e-6);
        assert!(result.x[1].abs() < 1e-6);
    }

    #[test]
    fn newton_shifted_quadratic() {
        // f(x) = (x - 3)² → minimum at x = 3
        let inner = LoweredOp::Sub(Box::new(var(0)), Box::new(c(3.0)));
        let cost = LoweredOp::Mul(Box::new(inner.clone()), Box::new(inner));
        let result = newton(&cost, &[0.0], 10, 1e-8).expect("converge");
        assert!(result.converged);
        assert!((result.x[0] - 3.0).abs() < 1e-6, "x = {}", result.x[0]);
    }

    #[test]
    fn newton_dim_mismatch_returns_err() {
        let cost = LoweredOp::Mul(Box::new(var(0)), Box::new(var(0)));
        let result = newton(&cost, &[], 10, 1e-8);
        assert!(matches!(
            result,
            Err(SymbolicNewtonError::DimMismatch { .. })
        ));
    }

    #[test]
    fn solve_linear_2x2() {
        // [2 1; 1 3] · x = [5; 10]  → x = [1; 3]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 10.0];
        let x = solve_linear(&a, &b).expect("solve");
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn solve_linear_singular_returns_err() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]]; // rank 1
        let b = vec![3.0, 6.0];
        assert!(matches!(
            solve_linear(&a, &b),
            Err(SymbolicNewtonError::SingularHessian)
        ));
    }
}
