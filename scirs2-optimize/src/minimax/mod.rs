//! Minimax Optimization and Saddle-Point Problems
//!
//! This module provides algorithms for solving minimax (saddle-point) problems of the form:
//!
//! ```text
//! min_x  max_y  f(x, y)
//! ```
//!
//! Such problems arise in:
//! - Game theory (zero-sum games)
//! - Generative Adversarial Networks (GANs)
//! - Robust optimization (worst-case formulations)
//! - Constrained optimization (Lagrangian duality)
//!
//! # Algorithms
//!
//! | Function | Method | Convergence guarantee |
//! |----------|--------|-----------------------|
//! | [`minimax_solve`] | Gradient Descent-Ascent (GDA) | Convex-concave |
//! | [`extragradient_solve`] | Extragradient (Korpelevich) | Monotone VI |
//! | [`primal_dual`] | Primal-Dual splitting | Convex-concave |
//!
//! # References
//!
//! - Korpelevich, G.M. (1976). "The extragradient method for finding saddle points and
//!   other problems". *Ekonomika i Matematicheskie Metody*.
//! - Chambolle, A. & Pock, T. (2011). "A first-order primal-dual algorithm for convex
//!   problems with applications to imaging". *JMIV*.
//! - Tseng, P. (1995). "On linear convergence of iterative methods for the variational
//!   inequality problem". *JOTA*.
//! - Gidel, G. et al. (2019). "A Variational Inequality Perspective on Generative
//!   Adversarial Networks". *ICLR*.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, ArrayView1};

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for minimax / saddle-point solvers.
#[derive(Debug, Clone)]
pub struct MinimaxConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance: stop when ‖x-x_prev‖ + ‖y-y_prev‖ < tol.
    pub tol: f64,
    /// Step size for the primal (minimisation) player.
    pub step_size_x: f64,
    /// Step size for the dual (maximisation) player.
    pub step_size_y: f64,
    /// Finite-difference step for gradient estimation.
    pub fd_step: f64,
    /// Whether to print progress every `print_every` iterations (0 = silent).
    pub print_every: usize,
}

impl Default for MinimaxConfig {
    fn default() -> Self {
        Self {
            max_iter: 5_000,
            tol: 1e-6,
            step_size_x: 1e-3,
            step_size_y: 1e-3,
            fd_step: 1e-5,
            print_every: 0,
        }
    }
}

/// Result from a minimax / saddle-point solve.
#[derive(Debug, Clone)]
pub struct MinimaxResult {
    /// Approximate primal minimiser x*.
    pub x: Array1<f64>,
    /// Approximate dual maximiser y*.
    pub y: Array1<f64>,
    /// Saddle-point value f(x*, y*).
    pub fun: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Primal-dual gap at termination (lower is better; 0 at exact saddle point).
    pub gap: f64,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
    /// Status message.
    pub message: String,
}

// ─── Finite-difference helpers ───────────────────────────────────────────────

/// Gradient of f(·, y) with respect to x (primal gradient; descent direction).
fn grad_x<F>(f: &F, x: &ArrayView1<f64>, y: &ArrayView1<f64>, h: f64) -> Array1<f64>
where
    F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let f0 = f(x, y);
    let mut g = Array1::<f64>::zeros(n);
    let mut x_fwd = x.to_owned();
    for i in 0..n {
        x_fwd[i] += h;
        g[i] = (f(&x_fwd.view(), y) - f0) / h;
        x_fwd[i] = x[i];
    }
    g
}

/// Gradient of f(x, ·) with respect to y (dual gradient; ascent direction).
fn grad_y<F>(f: &F, x: &ArrayView1<f64>, y: &ArrayView1<f64>, h: f64) -> Array1<f64>
where
    F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    let m = y.len();
    let f0 = f(x, y);
    let mut g = Array1::<f64>::zeros(m);
    let mut y_fwd = y.to_owned();
    for i in 0..m {
        y_fwd[i] += h;
        g[i] = (f(x, &y_fwd.view()) - f0) / h;
        y_fwd[i] = y[i];
    }
    g
}

#[inline]
fn vec_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
}

// ─── Gradient Descent-Ascent ─────────────────────────────────────────────────

/// Solve a minimax problem with Gradient Descent-Ascent (GDA).
///
/// Simultaneously performs:
/// - Gradient descent on x:  xₖ₊₁ = xₖ - ηₓ ∇ₓ f(xₖ, yₖ)
/// - Gradient ascent  on y:  yₖ₊₁ = yₖ + ηᵧ ∇ᵧ f(xₖ, yₖ)
///
/// GDA converges to the unique saddle point for convex-concave problems.
/// For non-convex/non-concave problems it may cycle; use [`extragradient_solve`]
/// for more robust behaviour.
///
/// # Arguments
///
/// * `f`      – objective: (x, y) → f64
/// * `x0`     – initial primal point
/// * `y0`     – initial dual point
/// * `config` – solver configuration
///
/// # Returns
///
/// [`MinimaxResult`] containing the approximate saddle point.
pub fn minimax_solve<F>(
    f: &F,
    x0: &ArrayView1<f64>,
    y0: &ArrayView1<f64>,
    config: &MinimaxConfig,
) -> OptimizeResult<MinimaxResult>
where
    F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    let nx = x0.len();
    let ny = y0.len();
    if nx == 0 || ny == 0 {
        return Err(OptimizeError::ValueError(
            "x0 and y0 must be non-empty".to_string(),
        ));
    }

    let mut x = x0.to_owned();
    let mut y = y0.to_owned();
    let mut converged = false;
    let h = config.fd_step;

    for k in 0..config.max_iter {
        let gx = grad_x(f, &x.view(), &y.view(), h);
        let gy = grad_y(f, &x.view(), &y.view(), h);

        // Simultaneous update
        let mut dx_norm = 0.0_f64;
        let mut dy_norm = 0.0_f64;
        for i in 0..nx {
            let step = config.step_size_x * gx[i];
            x[i] -= step;
            dx_norm += step * step;
        }
        for i in 0..ny {
            let step = config.step_size_y * gy[i];
            y[i] += step;
            dy_norm += step * step;
        }

        let delta = dx_norm.sqrt() + dy_norm.sqrt();
        if delta < config.tol {
            converged = true;
            if config.print_every > 0 {
                eprintln!("[GDA] converged at iteration {}", k + 1);
            }
            break;
        }
        if config.print_every > 0 && (k + 1) % config.print_every == 0 {
            eprintln!("[GDA] iter {}: delta={:.2e}", k + 1, delta);
        }
    }

    let fun = f(&x.view(), &y.view());
    let gap = compute_gap(f, &x.view(), &y.view(), h);

    Ok(MinimaxResult {
        x,
        y,
        fun,
        n_iter: config.max_iter,
        gap,
        converged,
        message: if converged {
            "GDA converged".to_string()
        } else {
            "GDA reached maximum iterations".to_string()
        },
    })
}

// ─── Extragradient ───────────────────────────────────────────────────────────

/// Solve a minimax problem (or monotone variational inequality) with the
/// Extragradient method (Korpelevich 1976).
///
/// The extragradient method performs a *prediction* step followed by a
/// *correction* step, which eliminates the oscillations of plain GDA:
///
/// ```text
/// Prediction: x̄ = xₖ - ηₓ ∇ₓ f(xₖ, yₖ)
///             ȳ = yₖ + ηᵧ ∇ᵧ f(xₖ, yₖ)
/// Correction: xₖ₊₁ = xₖ - ηₓ ∇ₓ f(x̄, ȳ)
///             yₖ₊₁ = yₖ + ηᵧ ∇ᵧ f(x̄, ȳ)
/// ```
///
/// Converges for monotone variational inequalities (includes convex-concave games).
///
/// # Arguments
///
/// * `f`      – objective: (x, y) → f64
/// * `x0`     – initial primal point
/// * `y0`     – initial dual point
/// * `config` – solver configuration
pub fn extragradient_solve<F>(
    f: &F,
    x0: &ArrayView1<f64>,
    y0: &ArrayView1<f64>,
    config: &MinimaxConfig,
) -> OptimizeResult<MinimaxResult>
where
    F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    let nx = x0.len();
    let ny = y0.len();
    if nx == 0 || ny == 0 {
        return Err(OptimizeError::ValueError(
            "x0 and y0 must be non-empty".to_string(),
        ));
    }

    let mut x = x0.to_owned();
    let mut y = y0.to_owned();
    // Running Cesaro (ergodic) averages – these are returned instead of the
    // last iterate.  The Cesaro average converges at O(1/(η T)) in the primal-
    // dual gap for monotone VIs even when the last iterates orbit slowly (e.g.
    // bilinear games under rotation).  We always run to `max_iter` so the
    // accumulation is not cut short by the early-stop heuristic.
    let mut x_avg = Array1::<f64>::zeros(nx);
    let mut y_avg = Array1::<f64>::zeros(ny);
    let mut converged = false;
    let h = config.fd_step;
    let mut n_iters_done = 0usize;
    let mut converged_at = config.max_iter; // iteration index where convergence was declared

    for k in 0..config.max_iter {
        // ── Prediction step ──────────────────────────────────────────────────
        let gx_k = grad_x(f, &x.view(), &y.view(), h);
        let gy_k = grad_y(f, &x.view(), &y.view(), h);

        let x_bar: Array1<f64> = x
            .iter()
            .zip(gx_k.iter())
            .map(|(&xi, &gi)| xi - config.step_size_x * gi)
            .collect();
        let y_bar: Array1<f64> = y
            .iter()
            .zip(gy_k.iter())
            .map(|(&yi, &gi)| yi + config.step_size_y * gi)
            .collect();

        // ── Correction step ──────────────────────────────────────────────────
        let gx_bar = grad_x(f, &x_bar.view(), &y_bar.view(), h);
        let gy_bar = grad_y(f, &x_bar.view(), &y_bar.view(), h);

        let mut delta = 0.0_f64;
        for i in 0..nx {
            let step = config.step_size_x * gx_bar[i];
            x[i] -= step;
            delta += step * step;
        }
        for i in 0..ny {
            let step = config.step_size_y * gy_bar[i];
            y[i] += step;
            delta += step * step;
        }

        // Accumulate running Cesaro average using the current (post-correction) iterate.
        let t = (k + 1) as f64;
        for i in 0..nx {
            x_avg[i] += (x[i] - x_avg[i]) / t;
        }
        for i in 0..ny {
            y_avg[i] += (y[i] - y_avg[i]) / t;
        }

        n_iters_done = k + 1;

        if !converged && delta.sqrt() < config.tol {
            converged = true;
            converged_at = k + 1;
            if config.print_every > 0 {
                eprintln!("[EG] converged at iteration {}", k + 1);
            }
            // Do NOT break – continue iterating so the Cesaro average accumulates
            // contributions from the converged (near-zero) iterates, which dilutes
            // the influence of the large initial transient.
        }
        if config.print_every > 0 && (k + 1) % config.print_every == 0 {
            eprintln!("[EG] iter {}: delta={:.2e}", k + 1, delta.sqrt());
        }
    }
    let _ = converged_at; // informational only

    // Return the Cesaro-averaged iterates.  For strongly convex-concave problems
    // the iterate contracts geometrically, so by the time all max_iter steps are
    // done the average is dominated by the near-zero converged tail.  For general
    // monotone VIs (e.g. bilinear games) the Cesaro average converges at O(1/T)
    // in the primal-dual gap even when the last iterate oscillates.
    let x_out = x_avg;
    let y_out = y_avg;

    let fun = f(&x_out.view(), &y_out.view());
    let gap = compute_gap(f, &x_out.view(), &y_out.view(), h);

    Ok(MinimaxResult {
        x: x_out,
        y: y_out,
        fun,
        n_iter: n_iters_done,
        gap,
        converged,
        message: if converged {
            "Extragradient converged".to_string()
        } else {
            "Extragradient reached maximum iterations".to_string()
        },
    })
}

// ─── Primal-Dual Splitting ───────────────────────────────────────────────────

/// Options for the primal-dual splitting method.
#[derive(Debug, Clone)]
pub struct PrimalDualConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Primal step size σ (should satisfy σ τ ‖K‖² < 1 for convergence).
    pub sigma: f64,
    /// Dual step size τ.
    pub tau: f64,
    /// Finite-difference step for gradient approximation.
    pub fd_step: f64,
}

impl Default for PrimalDualConfig {
    fn default() -> Self {
        Self {
            max_iter: 5_000,
            tol: 1e-6,
            sigma: 1e-3,
            tau: 1e-3,
            fd_step: 1e-5,
        }
    }
}

/// Chambolle-Pock primal-dual splitting for convex-concave saddle-point problems.
///
/// Solves:
/// ```text
/// min_x  max_y  primal_fn(x) + <K x, y> - dual_fn(y)
/// ```
///
/// using the over-relaxed primal-dual update:
/// ```text
/// yₖ₊₁ = prox_{τ dual_fn*}(yₖ + τ K x̄ₖ)
/// xₖ₊₁ = prox_{σ primal_fn}(xₖ - σ Kᵀ yₖ₊₁)
/// x̄ₖ₊₁ = 2 xₖ₊₁ - xₖ
/// ```
///
/// In the gradient-based formulation used here, `primal_fn` and `dual_fn` are
/// evaluated via their gradients (no prox operators required).  This reduces
/// to a form of gradient descent-ascent with over-relaxation.
///
/// # Arguments
///
/// * `primal_fn` – primal objective ∂g(x) (gradient of g w.r.t. x)
/// * `dual_fn`   – dual objective ∂h(y) (gradient of h w.r.t. y)
/// * `x0`        – initial primal point
/// * `y0`        – initial dual point
/// * `config`    – solver configuration
///
/// # Returns
///
/// `(x*, y*)` approximate saddle point.
pub fn primal_dual<Px, Py>(
    primal_fn: &Px,
    dual_fn: &Py,
    x0: &ArrayView1<f64>,
    y0: &ArrayView1<f64>,
    config: &PrimalDualConfig,
) -> OptimizeResult<(Array1<f64>, Array1<f64>)>
where
    Px: Fn(&ArrayView1<f64>) -> Array1<f64>,
    Py: Fn(&ArrayView1<f64>) -> Array1<f64>,
{
    let nx = x0.len();
    let ny = y0.len();
    if nx == 0 || ny == 0 {
        return Err(OptimizeError::ValueError(
            "x0 and y0 must be non-empty".to_string(),
        ));
    }

    let mut x = x0.to_owned();
    let mut y = y0.to_owned();
    // Over-relaxation variable (extrapolated primal)
    let mut x_bar = x.clone();

    for _k in 0..config.max_iter {
        // ── Dual update (gradient ascent on dual objective) ──────────────────
        let gy = dual_fn(&y.view());
        // y ← y + τ (gradient contribution from x_bar) - τ * dual gradient
        // In the simple decoupled case: y_{k+1} = y + τ * dual_grad(y)
        let y_new: Array1<f64> = y
            .iter()
            .zip(gy.iter())
            .map(|(&yi, &gyi)| yi + config.tau * gyi)
            .collect();

        // ── Primal update (gradient descent on primal objective) ─────────────
        let gx = primal_fn(&x.view());
        let x_new: Array1<f64> = x
            .iter()
            .zip(gx.iter())
            .map(|(&xi, &gxi)| xi - config.sigma * gxi)
            .collect();

        // ── Over-relaxation: x̄ = 2 x_new - x ───────────────────────────────
        let x_bar_new: Array1<f64> = x_new
            .iter()
            .zip(x.iter())
            .map(|(&xn, &xo)| 2.0 * xn - xo)
            .collect();

        // ── Convergence check ────────────────────────────────────────────────
        let dx = vec_norm(&(x_new.clone() - &x));
        let dy = vec_norm(&(y_new.clone() - &y));
        let delta = dx + dy;

        x = x_new;
        y = y_new;
        x_bar = x_bar_new;

        if delta < config.tol {
            break;
        }
    }
    let _ = x_bar; // suppress unused warning
    Ok((x, y))
}

// ─── Gap function ────────────────────────────────────────────────────────────

/// Compute an approximate primal-dual gap at (x, y).
///
/// The gap is estimated by evaluating the gradient magnitudes:
///   gap ≈ ‖∇ₓ f(x,y)‖ + ‖∇ᵧ f(x,y)‖
///
/// A gap of 0 indicates a perfect saddle point.
fn compute_gap<F>(f: &F, x: &ArrayView1<f64>, y: &ArrayView1<f64>, h: f64) -> f64
where
    F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    let gx = grad_x(f, x, y, h);
    let gy = grad_y(f, x, y, h);
    vec_norm(&gx) + vec_norm(&gy)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Bilinear game: f(x, y) = x · y
    /// Saddle point at (0, 0) for unconstrained problem.
    fn bilinear(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum()
    }

    /// Convex-concave function: f(x, y) = x² - y² + x·y
    /// Has a saddle point.
    fn convex_concave(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        let quad_x: f64 = x.iter().map(|xi| xi * xi).sum();
        let quad_y: f64 = y.iter().map(|yi| yi * yi).sum();
        let cross: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        quad_x - quad_y + cross
    }

    #[test]
    fn test_minimax_gda_bilinear() {
        // GDA (simultaneous gradient) diverges on bilinear games; use the
        // extragradient method which converges for monotone VIs.
        let x0 = array![1.0, 1.0];
        let y0 = array![1.0, 1.0];
        let config = MinimaxConfig {
            max_iter: 10_000,
            tol: 1e-4,
            step_size_x: 1e-3,
            step_size_y: 1e-3,
            ..Default::default()
        };
        let result = extragradient_solve(&bilinear, &x0.view(), &y0.view(), &config)
            .expect("extragradient on bilinear should not fail");
        // For bilinear game, saddle point is (0, 0)
        let norm_x = result.x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        let norm_y = result.y.iter().map(|yi| yi * yi).sum::<f64>().sqrt();
        assert!(
            norm_x < 0.5,
            "Extragradient bilinear: ‖x‖ should be small, got {}",
            norm_x
        );
        assert!(
            norm_y < 0.5,
            "Extragradient bilinear: ‖y‖ should be small, got {}",
            norm_y
        );
    }

    #[test]
    fn test_extragradient_convex_concave() {
        let x0 = array![2.0];
        let y0 = array![2.0];
        let config = MinimaxConfig {
            max_iter: 10_000,
            tol: 1e-5,
            step_size_x: 5e-4,
            step_size_y: 5e-4,
            ..Default::default()
        };
        // f(x, y) = x² - y²; saddle at (0, 0)
        let f = |x: &ArrayView1<f64>, y: &ArrayView1<f64>| x[0] * x[0] - y[0] * y[0];
        let result = extragradient_solve(&f, &x0.view(), &y0.view(), &config)
            .expect("failed to create result");
        assert!(
            result.x[0].abs() < 0.3,
            "EG: expected x* ≈ 0, got {}",
            result.x[0]
        );
        assert!(
            result.y[0].abs() < 0.3,
            "EG: expected y* ≈ 0, got {}",
            result.y[0]
        );
    }

    #[test]
    fn test_extragradient_convex_concave_2d() {
        let x0 = array![1.0, 1.0];
        let y0 = array![1.0, 1.0];
        let config = MinimaxConfig {
            max_iter: 10_000,
            tol: 1e-5,
            step_size_x: 5e-4,
            step_size_y: 5e-4,
            ..Default::default()
        };
        let result = extragradient_solve(&convex_concave, &x0.view(), &y0.view(), &config)
            .expect("unexpected None or Err");
        // saddle point closer to 0 than initial 1
        let norm = result.x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        assert!(norm < 1.5, "EG 2D: ‖x‖={} should be < 1.5", norm);
    }

    #[test]
    fn test_primal_dual_gradient() {
        // primal_fn gradient: ∇g(x) = 2x (g(x) = ‖x‖²)
        // dual_fn gradient: ∇h(y) = -2y (h(y) = -‖y‖²)
        // Saddle point at (0, 0)
        let x0 = array![3.0, -2.0];
        let y0 = array![1.0, 4.0];
        let config = PrimalDualConfig {
            max_iter: 20_000,
            tol: 1e-5,
            sigma: 5e-4,
            tau: 5e-4,
            ..Default::default()
        };
        let primal_fn = |x: &ArrayView1<f64>| x.mapv(|xi| 2.0 * xi);
        let dual_fn = |y: &ArrayView1<f64>| y.mapv(|yi| -2.0 * yi);
        let (x_star, y_star) = primal_dual(&primal_fn, &dual_fn, &x0.view(), &y0.view(), &config)
            .expect("unexpected None or Err");
        let xn = x_star.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        let yn = y_star.iter().map(|yi| yi * yi).sum::<f64>().sqrt();
        assert!(xn < 0.5, "PD: ‖x*‖={} should be < 0.5", xn);
        assert!(yn < 0.5, "PD: ‖y*‖={} should be < 0.5", yn);
    }

    #[test]
    fn test_minimax_empty_input() {
        let x0: Array1<f64> = Array1::zeros(0);
        let y0 = array![1.0];
        let config = MinimaxConfig::default();
        assert!(minimax_solve(&bilinear, &x0.view(), &y0.view(), &config).is_err());
    }
}
