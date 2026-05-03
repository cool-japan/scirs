//! Classical stochastic approximation algorithms.
//!
//! This submodule implements the three canonical stochastic approximation (SA)
//! methods from the 1950s–1960s together with a modern SPSA variant:
//!
//! | Algorithm | Reference | Use case |
//! |-----------|-----------|----------|
//! | Robbins-Monro | Robbins & Monro (1951) | Root-finding under noise |
//! | Kiefer-Wolfowitz | Kiefer & Wolfowitz (1952) | Gradient-free stochastic minimisation |
//! | SPSA | Spall (1992) | High-dimensional gradient-free SA |
//!
//! # Notation
//!
//! - xₖ  : current iterate
//! - aₖ  : gain sequence for update step (must satisfy Σ aₖ = ∞, Σ aₖ² < ∞)
//! - cₖ  : gain sequence for finite-difference width (must → 0)

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, ArrayView1};

// ─── Robbins-Monro ───────────────────────────────────────────────────────────

/// Result from Robbins-Monro root finding.
#[derive(Debug, Clone)]
pub struct RobbinsMonroResult {
    /// Approximate root: θ* such that M(θ*) ≈ 0.
    pub x: Array1<f64>,
    /// Final residual ‖M(xₖ)‖.
    pub residual: f64,
    /// Number of iterations.
    pub n_iter: usize,
    /// Whether tolerance was met.
    pub converged: bool,
}

/// Options for the Robbins-Monro algorithm.
#[derive(Debug, Clone)]
pub struct RobbinsMonroOptions {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on ‖update step‖.
    pub tol: f64,
    /// Exponent α in the gain aₖ = a / kᵅ.  Standard choice: α = 1.0.
    pub alpha: f64,
    /// Scale a in aₖ = a / kᵅ.
    pub a: f64,
}

impl Default for RobbinsMonroOptions {
    fn default() -> Self {
        Self {
            max_iter: 10_000,
            tol: 1e-6,
            alpha: 1.0,
            a: 1.0,
        }
    }
}

/// Robbins-Monro stochastic root-finding algorithm.
///
/// Finds θ* such that M(θ*) = 0, where M is a noisy mapping (possibly
/// the gradient of an expected loss).  The update rule is:
///
/// θₖ₊₁ = θₖ - aₖ · M(θₖ)
///
/// with gain aₖ = a / k^α.
///
/// # Arguments
///
/// * `m`    – noisy mapping M: θ → ℝⁿ (should satisfy E[M(θ)] ≈ ∇L(θ))
/// * `x0`   – initial point
/// * `opts` – algorithm options
pub fn robbins_monro<M>(
    m: &mut M,
    x0: &ArrayView1<f64>,
    opts: &RobbinsMonroOptions,
) -> OptimizeResult<RobbinsMonroResult>
where
    M: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }

    let mut x = x0.to_owned();
    let mut converged = false;
    let mut residual = f64::INFINITY;

    for k in 1..=opts.max_iter {
        let mk = m(&x.view());
        if mk.len() != n {
            return Err(OptimizeError::ValueError(format!(
                "M returned length {} but x has length {}",
                mk.len(),
                n
            )));
        }
        let ak = opts.a / (k as f64).powf(opts.alpha);
        let mut step_norm = 0.0_f64;
        for i in 0..n {
            let step = ak * mk[i];
            x[i] -= step;
            step_norm += step * step;
        }
        residual = step_norm.sqrt();
        if residual < opts.tol {
            converged = true;
            residual = mk.iter().map(|v| v * v).sum::<f64>().sqrt();
            return Ok(RobbinsMonroResult {
                x,
                residual,
                n_iter: k,
                converged,
            });
        }
    }

    // Final residual
    let mk_final = m(&x.view());
    residual = mk_final.iter().map(|v| v * v).sum::<f64>().sqrt();

    Ok(RobbinsMonroResult {
        x,
        residual,
        n_iter: opts.max_iter,
        converged,
    })
}

// ─── Kiefer-Wolfowitz ────────────────────────────────────────────────────────

/// Result from the Kiefer-Wolfowitz algorithm.
#[derive(Debug, Clone)]
pub struct KieferWolfowitzResult {
    /// Approximate minimiser.
    pub x: Array1<f64>,
    /// Function value at x.
    pub fun: f64,
    /// Number of iterations.
    pub n_iter: usize,
    /// Whether tolerance was met.
    pub converged: bool,
}

/// Options for the Kiefer-Wolfowitz algorithm.
#[derive(Debug, Clone)]
pub struct KieferWolfowitzOptions {
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance (step norm).
    pub tol: f64,
    /// Exponent α in aₖ = a / kᵅ.  Must satisfy α ∈ (1/2, 1].
    pub alpha: f64,
    /// Exponent γ in cₖ = c / kᵞ.  Must satisfy γ ∈ (0, 1/6] for unbiased gradients.
    pub gamma: f64,
    /// Scale constant a.
    pub a: f64,
    /// Scale constant c (initial finite-difference width).
    pub c: f64,
}

impl Default for KieferWolfowitzOptions {
    fn default() -> Self {
        Self {
            max_iter: 10_000,
            tol: 1e-6,
            alpha: 0.602,
            gamma: 0.101,
            a: 0.1,
            c: 0.1,
        }
    }
}

/// Kiefer-Wolfowitz gradient-free stochastic approximation.
///
/// Minimises E[L(x, ξ)] using only noisy function evaluations (no gradients).
/// The finite-difference gradient estimate in dimension i is:
///
/// ĝᵢ = (L(x + cₖ eᵢ) - L(x - cₖ eᵢ)) / (2 cₖ)
///
/// followed by the update x ← x - aₖ ĝ.
///
/// # Arguments
///
/// * `loss` – noisy loss function (x) → f64 (internally uses two evaluations per dimension per step)
/// * `x0`   – initial point
/// * `opts` – algorithm options
pub fn kiefer_wolfowitz<L>(
    loss: &mut L,
    x0: &ArrayView1<f64>,
    opts: &KieferWolfowitzOptions,
) -> OptimizeResult<KieferWolfowitzResult>
where
    L: FnMut(&ArrayView1<f64>) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }

    let mut x = x0.to_owned();
    let mut converged = false;

    for k in 1..=opts.max_iter {
        let ak = opts.a / (k as f64).powf(opts.alpha);
        let ck = opts.c / (k as f64).powf(opts.gamma);

        // Finite-difference gradient
        let mut grad = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut x_fwd = x.clone();
            let mut x_bwd = x.clone();
            x_fwd[i] += ck;
            x_bwd[i] -= ck;
            grad[i] = (loss(&x_fwd.view()) - loss(&x_bwd.view())) / (2.0 * ck);
        }

        let mut step_norm = 0.0_f64;
        for i in 0..n {
            let step = ak * grad[i];
            x[i] -= step;
            step_norm += step * step;
        }

        if step_norm.sqrt() < opts.tol {
            converged = true;
            let fun = loss(&x.view());
            return Ok(KieferWolfowitzResult {
                x,
                fun,
                n_iter: k,
                converged,
            });
        }
    }

    let fun = loss(&x.view());
    Ok(KieferWolfowitzResult {
        x,
        fun,
        n_iter: opts.max_iter,
        converged,
    })
}

// ─── SPSA ────────────────────────────────────────────────────────────────────

/// Options for the SPSA optimizer.
#[derive(Debug, Clone)]
pub struct SpsaOptions {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on ‖step‖.
    pub tol: f64,
    /// Exponent α in aₖ = a / (A + k)^α.
    pub alpha: f64,
    /// Exponent γ in cₖ = c / k^γ.
    pub gamma: f64,
    /// Scale a.
    pub a: f64,
    /// Stability constant A (typically ≈ 0.1 * max_iter).
    pub big_a: f64,
    /// Finite-difference constant c.
    pub c: f64,
}

impl Default for SpsaOptions {
    fn default() -> Self {
        Self {
            max_iter: 5_000,
            tol: 1e-6,
            alpha: 0.602,
            gamma: 0.101,
            a: 0.1,
            big_a: 100.0,
            c: 0.1,
        }
    }
}

/// Result from the SPSA algorithm.
#[derive(Debug, Clone)]
pub struct SpsaResult {
    /// Approximate minimiser.
    pub x: Array1<f64>,
    /// Function value at x.
    pub fun: f64,
    /// Number of iterations.
    pub n_iter: usize,
    /// Whether tolerance was met.
    pub converged: bool,
}

/// Compute one SPSA gradient-estimate step.
///
/// The simultaneous perturbation direction Δ is sampled from {±1}ⁿ (Rademacher).
/// The gradient estimate is:
///
/// ĝ(x) = [f(x + cₖ Δ) - f(x - cₖ Δ)] / (2 cₖ) * (1/Δᵢ)  component-wise.
///
/// Returns the updated x after one SPSA step.
///
/// # Arguments
///
/// * `f`     – noisy objective function
/// * `x`     – current point (modified in place)
/// * `k`     – current iteration number (1-indexed)
/// * `opts`  – SPSA options
/// * `rng`   – mutable u64 state for the Rademacher perturbation (LCG)
pub fn spsa_step<F>(
    f: &mut F,
    x: &mut Array1<f64>,
    k: usize,
    opts: &SpsaOptions,
    rng_state: &mut u64,
) -> f64
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let ak = opts.a / (opts.big_a + k as f64).powf(opts.alpha);
    let ck = opts.c / (k as f64).powf(opts.gamma);

    // Draw Rademacher perturbation Δ ∈ {-1, +1}ⁿ using LCG
    let mut delta = Array1::<f64>::zeros(n);
    for i in 0..n {
        *rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        delta[i] = if (*rng_state >> 63) == 0 { 1.0 } else { -1.0 };
    }

    // Two-sided function evaluations
    let x_fwd: Array1<f64> = x
        .iter()
        .zip(delta.iter())
        .map(|(&xi, &di)| xi + ck * di)
        .collect();
    let x_bwd: Array1<f64> = x
        .iter()
        .zip(delta.iter())
        .map(|(&xi, &di)| xi - ck * di)
        .collect();
    let f_fwd = f(&x_fwd.view());
    let f_bwd = f(&x_bwd.view());

    let diff = (f_fwd - f_bwd) / (2.0 * ck);

    // Update: x ← x - aₖ * ĝ  where ĝᵢ = diff / Δᵢ
    let mut step_sq = 0.0_f64;
    for i in 0..n {
        let gi = diff / delta[i]; // Δᵢ ∈ {±1} so 1/Δᵢ = Δᵢ
        let step = ak * gi;
        x[i] -= step;
        step_sq += step * step;
    }
    step_sq.sqrt()
}

/// Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.
///
/// SPSA uses only two function evaluations per iteration (regardless of dimension n),
/// making it very efficient for high-dimensional black-box minimisation.
///
/// # Arguments
///
/// * `f`    – noisy objective (minimised)
/// * `x0`   – starting point
/// * `opts` – SPSA options
pub fn spsa_minimize<F>(
    f: &mut F,
    x0: &ArrayView1<f64>,
    opts: &SpsaOptions,
) -> OptimizeResult<SpsaResult>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    if x0.is_empty() {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }

    let mut x = x0.to_owned();
    let mut rng_state: u64 = 12345678901234567;
    let mut converged = false;

    for k in 1..=opts.max_iter {
        let step_norm = spsa_step(f, &mut x, k, opts, &mut rng_state);
        if step_norm < opts.tol {
            converged = true;
            let fun = f(&x.view());
            return Ok(SpsaResult {
                x,
                fun,
                n_iter: k,
                converged,
            });
        }
    }

    let fun = f(&x.view());
    Ok(SpsaResult {
        x,
        fun,
        n_iter: opts.max_iter,
        converged,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_robbins_monro_linear() {
        // M(x) = x - 2 (root at x=2)
        let mut m = |x: &ArrayView1<f64>| array![x[0] - 2.0];
        let x0 = array![0.0];
        let opts = RobbinsMonroOptions {
            max_iter: 50_000,
            tol: 1e-4,
            a: 1.0,
            alpha: 1.0,
        };
        let res = robbins_monro(&mut m, &x0.view(), &opts).expect("failed to create res");
        assert!(
            (res.x[0] - 2.0).abs() < 0.1,
            "expected x* ≈ 2.0, got {}",
            res.x[0]
        );
    }

    #[test]
    fn test_kiefer_wolfowitz_quadratic() {
        // L(x) = (x-3)²; minimiser at x=3
        let mut loss = |x: &ArrayView1<f64>| (x[0] - 3.0).powi(2);
        let x0 = array![0.0];
        let opts = KieferWolfowitzOptions {
            max_iter: 20_000,
            tol: 1e-5,
            ..Default::default()
        };
        let res = kiefer_wolfowitz(&mut loss, &x0.view(), &opts).expect("failed to create res");
        assert!(
            (res.x[0] - 3.0).abs() < 0.2,
            "expected x* ≈ 3.0, got {}",
            res.x[0]
        );
    }

    #[test]
    fn test_spsa_quadratic() {
        // f(x) = (x₀-1)² + (x₁-2)²; minimiser at (1, 2)
        let mut f = |x: &ArrayView1<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let x0 = array![0.0, 0.0];
        let opts = SpsaOptions {
            max_iter: 10_000,
            tol: 1e-5,
            a: 0.5,
            big_a: 50.0,
            c: 0.2,
            ..Default::default()
        };
        let res = spsa_minimize(&mut f, &x0.view(), &opts).expect("failed to create res");
        assert!(
            (res.x[0] - 1.0).abs() < 0.3,
            "expected x[0] ≈ 1.0, got {}",
            res.x[0]
        );
        assert!(
            (res.x[1] - 2.0).abs() < 0.3,
            "expected x[1] ≈ 2.0, got {}",
            res.x[1]
        );
    }
}
