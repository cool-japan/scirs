//! Variance Reduction Methods for Stochastic Gradient Descent
//!
//! Standard SGD suffers from variance due to mini-batch noise, which prevents
//! convergence to the exact minimiser.  Variance reduction methods achieve
//! *linear* convergence on strongly convex problems by periodically correcting
//! for the gradient bias.
//!
//! # Algorithms
//!
//! | Method | Reference | Gradient evals per pass |
//! |--------|-----------|--------------------------|
//! | SVRG   | Johnson & Zhang (2013) | 2n (snapshot + inner) |
//! | SARAH  | Nguyen et al. (2017)   | n + m·b               |
//! | SPIDER | Fang et al. (2018)     | n + m·b               |
//!
//! All three maintain a recursive gradient correction that cancels the noise
//! introduced by mini-batch subsampling.
//!
//! # References
//!
//! - Johnson, R. & Zhang, T. (2013). "Accelerating Stochastic Gradient Descent
//!   using Predictive Variance Reduction". *NeurIPS*.
//! - Nguyen, L.M. et al. (2017). "SARAH: A Novel Method for Machine Learning
//!   Problems Using Stochastic Recursive Gradient". *ICML*.
//! - Fang, C. et al. (2018). "SPIDER: Near-Optimal Non-Convex Optimization
//!   via Stochastic Path-Integrated Differential Estimator". *NeurIPS*.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, ArrayView1};

// ─── Shared helpers ──────────────────────────────────────────────────────────

/// Compute a finite-difference gradient estimate for a single data point.
#[inline]
fn finite_diff_grad<F>(
    f: &mut F,
    x: &ArrayView1<f64>,
    sample: &ArrayView1<f64>,
    h: f64,
) -> Array1<f64>
where
    F: FnMut(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let f0 = f(x, sample);
    let mut grad = Array1::<f64>::zeros(n);
    let mut x_fwd = x.to_owned();
    for i in 0..n {
        x_fwd[i] += h;
        grad[i] = (f(&x_fwd.view(), sample) - f0) / h;
        x_fwd[i] = x[i];
    }
    grad
}

/// Average gradient over a set of samples (full batch).
fn full_grad<F>(f: &mut F, x: &ArrayView1<f64>, samples: &[Array1<f64>], h: f64) -> Array1<f64>
where
    F: FnMut(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    let n = x.len();
    if samples.is_empty() {
        return Array1::zeros(n);
    }
    let mut avg = Array1::<f64>::zeros(n);
    for s in samples {
        let g = finite_diff_grad(f, x, &s.view(), h);
        for i in 0..n {
            avg[i] += g[i];
        }
    }
    let inv_m = 1.0 / samples.len() as f64;
    avg.mapv_inplace(|v| v * inv_m);
    avg
}

// ─── SVRG ────────────────────────────────────────────────────────────────────

/// Options for the SVRG optimizer.
#[derive(Debug, Clone)]
pub struct SvrgOptions {
    /// Number of outer iterations (snapshot updates).
    pub n_epochs: usize,
    /// Number of inner SGD steps per epoch.
    pub inner_steps: usize,
    /// Step size η.
    pub step_size: f64,
    /// Convergence tolerance (gradient norm of snapshot).
    pub tol: f64,
    /// Finite-difference step for gradient approximation.
    pub fd_step: f64,
}

impl Default for SvrgOptions {
    fn default() -> Self {
        Self {
            n_epochs: 50,
            inner_steps: 100,
            step_size: 1e-3,
            tol: 1e-6,
            fd_step: 1e-5,
        }
    }
}

/// Result from SVRG optimisation.
#[derive(Debug, Clone)]
pub struct SvrgResult {
    /// Approximate minimiser.
    pub x: Array1<f64>,
    /// Final full-gradient norm.
    pub grad_norm: f64,
    /// Total number of gradient evaluations.
    pub n_grad_evals: usize,
    /// Whether tolerance was met.
    pub converged: bool,
}

/// Stochastic Variance Reduced Gradient (SVRG) optimizer.
///
/// At the start of each epoch, a full-gradient μ̃ = ∇f(x̃) is computed at a
/// snapshot x̃.  Each inner step uses the variance-reduced direction:
///
/// v = ∇fᵢ(x) - ∇fᵢ(x̃) + μ̃
///
/// which has zero variance when x = x̃.
///
/// # Arguments
///
/// * `f`       – per-sample loss: (x, sample) → f64
/// * `x0`      – starting point
/// * `samples` – full dataset (each element is one sample parameter vector)
/// * `opts`    – SVRG options
pub fn svrg<F>(
    f: &mut F,
    x0: &ArrayView1<f64>,
    samples: &[Array1<f64>],
    opts: &SvrgOptions,
) -> OptimizeResult<SvrgResult>
where
    F: FnMut(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }
    if samples.is_empty() {
        return Err(OptimizeError::ValueError(
            "samples must be non-empty".to_string(),
        ));
    }

    let m = samples.len();
    let mut x = x0.to_owned();
    let mut converged = false;
    let mut total_evals: usize = 0;
    // LCG for sample selection
    let mut rng: u64 = 987654321;

    for _ in 0..opts.n_epochs {
        // Snapshot: x̃ ← x, compute full gradient μ̃
        let x_tilde = x.clone();
        let mu_tilde = full_grad(f, &x_tilde.view(), samples, opts.fd_step);
        total_evals += m * (n + 1);

        let grad_norm = mu_tilde.iter().map(|v| v * v).sum::<f64>().sqrt();
        if grad_norm < opts.tol {
            converged = true;
            return Ok(SvrgResult {
                x,
                grad_norm,
                n_grad_evals: total_evals,
                converged,
            });
        }

        // Inner loop
        for _ in 0..opts.inner_steps {
            // Pick random sample index via LCG
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (rng >> 33) as usize % m;
            let s = &samples[idx];

            let g_x = finite_diff_grad(f, &x.view(), &s.view(), opts.fd_step);
            let g_tilde = finite_diff_grad(f, &x_tilde.view(), &s.view(), opts.fd_step);
            total_evals += 2 * (n + 1);

            // SVRG direction: v = g(x) - g(x̃) + μ̃
            for i in 0..n {
                x[i] -= opts.step_size * (g_x[i] - g_tilde[i] + mu_tilde[i]);
            }
        }
    }

    let grad_norm = full_grad(f, &x.view(), samples, opts.fd_step)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();

    Ok(SvrgResult {
        x,
        grad_norm,
        n_grad_evals: total_evals,
        converged,
    })
}

// ─── SARAH ───────────────────────────────────────────────────────────────────

/// Options for the SARAH optimizer.
#[derive(Debug, Clone)]
pub struct SarahOptions {
    /// Number of outer iterations.
    pub n_outer: usize,
    /// Inner loop length m.
    pub inner_steps: usize,
    /// Step size η.
    pub step_size: f64,
    /// Convergence tolerance (full gradient norm).
    pub tol: f64,
    /// Finite-difference step.
    pub fd_step: f64,
}

impl Default for SarahOptions {
    fn default() -> Self {
        Self {
            n_outer: 50,
            inner_steps: 50,
            step_size: 1e-3,
            tol: 1e-6,
            fd_step: 1e-5,
        }
    }
}

/// Result from SARAH optimisation.
#[derive(Debug, Clone)]
pub struct SarahResult {
    /// Approximate minimiser.
    pub x: Array1<f64>,
    /// Final full-gradient norm.
    pub grad_norm: f64,
    /// Total gradient evaluations (approximate).
    pub n_grad_evals: usize,
    /// Whether tolerance was met.
    pub converged: bool,
}

/// StochAstic Recursive grAdient algoritHm (SARAH).
///
/// SARAH maintains a recursive gradient estimator:
///
///   v₀ = ∇f(x₀)   (full gradient)
///   vₜ = ∇fᵢₜ(xₜ) - ∇fᵢₜ(xₜ₋₁) + vₜ₋₁   (recursive update)
///   xₜ₊₁ = xₜ - η vₜ
///
/// This recursive estimator converges to the true gradient, enabling linear
/// convergence on strongly-convex problems.
///
/// # Arguments
///
/// * `f`       – per-sample loss: (x, sample) → f64
/// * `x0`      – starting point
/// * `samples` – full dataset
/// * `opts`    – SARAH options
pub fn sarah<F>(
    f: &mut F,
    x0: &ArrayView1<f64>,
    samples: &[Array1<f64>],
    opts: &SarahOptions,
) -> OptimizeResult<SarahResult>
where
    F: FnMut(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }
    if samples.is_empty() {
        return Err(OptimizeError::ValueError(
            "samples must be non-empty".to_string(),
        ));
    }

    let m = samples.len();
    let mut x = x0.to_owned();
    let mut converged = false;
    let mut total_evals: usize = 0;
    let mut rng: u64 = 11111111111111111;

    for _ in 0..opts.n_outer {
        // v₀ = full gradient at current x
        let mut v = full_grad(f, &x.view(), samples, opts.fd_step);
        total_evals += m * (n + 1);

        let g_norm = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt();
        if g_norm < opts.tol {
            converged = true;
            return Ok(SarahResult {
                x,
                grad_norm: g_norm,
                n_grad_evals: total_evals,
                converged,
            });
        }

        // x₀ of inner loop
        for i in 0..n {
            x[i] -= opts.step_size * v[i];
        }

        let mut x_prev = x.clone();

        for _ in 0..opts.inner_steps {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (rng >> 33) as usize % m;
            let s = &samples[idx];

            let g_curr = finite_diff_grad(f, &x.view(), &s.view(), opts.fd_step);
            let g_prev = finite_diff_grad(f, &x_prev.view(), &s.view(), opts.fd_step);
            total_evals += 2 * (n + 1);

            // Recursive update: vₜ = ∇fᵢ(x) - ∇fᵢ(x_prev) + v_{t-1}
            let v_new: Array1<f64> = g_curr
                .iter()
                .zip(g_prev.iter())
                .zip(v.iter())
                .map(|((&gc, &gp), &vp)| gc - gp + vp)
                .collect();

            x_prev = x.clone();
            for i in 0..n {
                x[i] -= opts.step_size * v_new[i];
            }
            v = v_new;
        }
    }

    let g_norm = full_grad(f, &x.view(), samples, opts.fd_step)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();

    Ok(SarahResult {
        x,
        grad_norm: g_norm,
        n_grad_evals: total_evals,
        converged,
    })
}

// ─── SPIDER ──────────────────────────────────────────────────────────────────

/// Options for the SPIDER optimizer.
#[derive(Debug, Clone)]
pub struct SpiderOptions {
    /// Number of outer iterations (full gradient recomputes).
    pub n_outer: usize,
    /// Inner steps per outer iteration.
    pub inner_steps: usize,
    /// Step size η.
    pub step_size: f64,
    /// Convergence tolerance (gradient estimator norm).
    pub tol: f64,
    /// Finite-difference step.
    pub fd_step: f64,
    /// Mini-batch size b for inner gradient updates.
    pub mini_batch: usize,
}

impl Default for SpiderOptions {
    fn default() -> Self {
        Self {
            n_outer: 30,
            inner_steps: 50,
            step_size: 5e-4,
            tol: 1e-6,
            fd_step: 1e-5,
            mini_batch: 4,
        }
    }
}

/// Result from SPIDER optimisation.
#[derive(Debug, Clone)]
pub struct SpiderResult {
    /// Approximate minimiser.
    pub x: Array1<f64>,
    /// Norm of the last gradient estimator.
    pub grad_norm: f64,
    /// Total gradient evaluations (approximate).
    pub n_grad_evals: usize,
    /// Whether tolerance was met.
    pub converged: bool,
}

/// SPIDER (Stochastic Path-Integrated Differential EstimatoR) optimizer.
///
/// SPIDER extends SARAH with mini-batch gradient differences, achieving
/// near-optimal oracle complexity for non-convex stochastic optimisation.
///
/// The estimator update:
///   vₜ = (1/b) Σᵢ∈Bₜ [∇fᵢ(xₜ) - ∇fᵢ(xₜ₋₁)] + vₜ₋₁
///
/// # Arguments
///
/// * `f`       – per-sample loss: (x, sample) → f64
/// * `x0`      – starting point
/// * `samples` – full dataset
/// * `opts`    – SPIDER options
pub fn spider<F>(
    f: &mut F,
    x0: &ArrayView1<f64>,
    samples: &[Array1<f64>],
    opts: &SpiderOptions,
) -> OptimizeResult<SpiderResult>
where
    F: FnMut(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }
    if samples.is_empty() {
        return Err(OptimizeError::ValueError(
            "samples must be non-empty".to_string(),
        ));
    }

    let m = samples.len();
    let b = opts.mini_batch.max(1).min(m);
    let mut x = x0.to_owned();
    let mut converged = false;
    let mut total_evals: usize = 0;
    let mut rng: u64 = 999999999999;

    for _ in 0..opts.n_outer {
        // Full gradient at start of outer epoch
        let mut v = full_grad(f, &x.view(), samples, opts.fd_step);
        total_evals += m * (n + 1);

        let g_norm = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt();
        if g_norm < opts.tol {
            converged = true;
            return Ok(SpiderResult {
                x,
                grad_norm: g_norm,
                n_grad_evals: total_evals,
                converged,
            });
        }

        // Descend with v₀
        for i in 0..n {
            x[i] -= opts.step_size * v[i];
        }

        let mut x_prev = x.clone();

        for _ in 0..opts.inner_steps {
            // Sample mini-batch B of size b
            let mut batch_indices = Vec::with_capacity(b);
            for _ in 0..b {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                batch_indices.push((rng >> 33) as usize % m);
            }

            // Mini-batch gradient difference
            let mut diff = Array1::<f64>::zeros(n);
            for &idx in &batch_indices {
                let s = &samples[idx];
                let g_curr = finite_diff_grad(f, &x.view(), &s.view(), opts.fd_step);
                let g_prev = finite_diff_grad(f, &x_prev.view(), &s.view(), opts.fd_step);
                total_evals += 2 * (n + 1);
                for i in 0..n {
                    diff[i] += (g_curr[i] - g_prev[i]) / b as f64;
                }
            }

            // Recursive estimator update
            let v_new: Array1<f64> = diff.iter().zip(v.iter()).map(|(&d, &vp)| d + vp).collect();
            x_prev = x.clone();
            for i in 0..n {
                x[i] -= opts.step_size * v_new[i];
            }
            v = v_new;

            let cur_norm = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt();
            if cur_norm < opts.tol {
                converged = true;
                return Ok(SpiderResult {
                    x,
                    grad_norm: cur_norm,
                    n_grad_evals: total_evals,
                    converged,
                });
            }
        }
    }

    let g_norm = v_norm_approx(&full_grad(f, &x.view(), samples, opts.fd_step));

    Ok(SpiderResult {
        x,
        grad_norm: g_norm,
        n_grad_evals: total_evals,
        converged,
    })
}

#[inline]
fn v_norm_approx(v: &Array1<f64>) -> f64 {
    v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Generate a simple quadratic dataset: f(x, ξ) = (x₀ - ξ₀)² + (x₁ - ξ₁)²
    /// Minimiser at E[ξ] = (1, 2).
    fn make_samples() -> Vec<Array1<f64>> {
        vec![
            array![0.9, 1.8],
            array![1.1, 2.2],
            array![1.0, 2.0],
            array![0.8, 1.9],
            array![1.2, 2.1],
            array![1.0, 2.0],
            array![0.95, 1.95],
            array![1.05, 2.05],
        ]
    }

    fn sample_loss(x: &ArrayView1<f64>, s: &ArrayView1<f64>) -> f64 {
        (x[0] - s[0]).powi(2) + (x[1] - s[1]).powi(2)
    }

    #[test]
    fn test_svrg_quadratic() {
        let samples = make_samples();
        let x0 = array![0.0, 0.0];
        let opts = SvrgOptions {
            n_epochs: 100,
            inner_steps: 50,
            step_size: 0.1,
            tol: 1e-4,
            fd_step: 1e-5,
        };
        let res = svrg(&mut |x, s| sample_loss(x, s), &x0.view(), &samples, &opts)
            .expect("failed to create res");
        assert!(
            (res.x[0] - 1.0).abs() < 0.3,
            "SVRG: expected x[0]≈1.0, got {}",
            res.x[0]
        );
        assert!(
            (res.x[1] - 2.0).abs() < 0.3,
            "SVRG: expected x[1]≈2.0, got {}",
            res.x[1]
        );
    }

    #[test]
    fn test_sarah_quadratic() {
        let samples = make_samples();
        let x0 = array![0.0, 0.0];
        let opts = SarahOptions {
            n_outer: 80,
            inner_steps: 30,
            step_size: 0.05,
            tol: 1e-4,
            fd_step: 1e-5,
        };
        let res = sarah(&mut |x, s| sample_loss(x, s), &x0.view(), &samples, &opts)
            .expect("failed to create res");
        assert!(
            (res.x[0] - 1.0).abs() < 0.3,
            "SARAH: expected x[0]≈1.0, got {}",
            res.x[0]
        );
        assert!(
            (res.x[1] - 2.0).abs() < 0.3,
            "SARAH: expected x[1]≈2.0, got {}",
            res.x[1]
        );
    }

    #[test]
    fn test_spider_quadratic() {
        let samples = make_samples();
        let x0 = array![0.0, 0.0];
        let opts = SpiderOptions {
            n_outer: 80,
            inner_steps: 30,
            step_size: 0.05,
            tol: 1e-4,
            fd_step: 1e-5,
            mini_batch: 2,
        };
        let res = spider(&mut |x, s| sample_loss(x, s), &x0.view(), &samples, &opts)
            .expect("failed to create res");
        assert!(
            (res.x[0] - 1.0).abs() < 0.4,
            "SPIDER: expected x[0]≈1.0, got {}",
            res.x[0]
        );
        assert!(
            (res.x[1] - 2.0).abs() < 0.4,
            "SPIDER: expected x[1]≈2.0, got {}",
            res.x[1]
        );
    }
}
