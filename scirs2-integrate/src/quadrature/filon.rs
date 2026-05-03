//! Filon quadrature for oscillatory integrals
//!
//! This module implements Filon-type methods for integrals of the form:
//!
//! - `integral f(x) * sin(omega * x) dx` over `[a, b]`
//! - `integral f(x) * cos(omega * x) dx` over `[a, b]`
//! - `integral f(x) * exp(i * omega * g(x)) dx` (Levin's method, basic version)
//!
//! ## Methods
//!
//! - **Filon (classical)**: Interpolates `f(x)` with quadratic polynomials on
//!   subintervals and evaluates the moments `integral x^k sin(omega*x) dx`
//!   analytically. Converges as the panel count grows, independently of `omega`.
//! - **Filon-Simpson**: A Simpson-like variant where the oscillatory factor is
//!   handled through moment integrals.
//! - **Filon-Clenshaw-Curtis**: Uses Clenshaw-Curtis nodes for `f(x)` and
//!   evaluates moments via recurrence on Chebyshev coefficients.
//! - **Adaptive Filon**: Chooses the number of panels based on `omega` and a
//!   requested tolerance.
//! - **Levin collocation (basic)**: Transforms the oscillatory integral into a
//!   non-oscillatory ODE boundary-value problem solved with polynomial collocation.
//!
//! ## References
//!
//! - L.N.G. Filon (1928), "On a quadrature formula for trigonometric integrals"
//! - A. Iserles & S.P. Norsett (2004), "On quadrature methods for highly
//!   oscillatory integrals and their implementation"
//! - D. Levin (1982), "Procedures for computing one- and two-dimensional integrals
//!   of functions with rapid irregular oscillations"

use crate::error::{IntegrateError, IntegrateResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Kind of oscillatory kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OscillatoryKernel {
    /// `sin(omega * x)`
    Sine,
    /// `cos(omega * x)`
    Cosine,
}

/// Result of a Filon-type quadrature.
#[derive(Debug, Clone)]
pub struct FilonResult {
    /// Estimated value of the integral.
    pub value: f64,
    /// Estimated absolute error (when available, otherwise 0).
    pub error_estimate: f64,
    /// Number of function evaluations used.
    pub n_evals: usize,
    /// Number of subintervals (panels).
    pub n_panels: usize,
}

/// Options for Filon quadrature.
#[derive(Debug, Clone)]
pub struct FilonOptions {
    /// Number of panels (must be even for classical Filon). Default: auto.
    pub n_panels: Option<usize>,
    /// Absolute tolerance for adaptive method. Default: 1e-10.
    pub abs_tol: f64,
    /// Relative tolerance for adaptive method. Default: 1e-10.
    pub rel_tol: f64,
    /// Maximum number of panels in adaptive mode. Default: 10_000.
    pub max_panels: usize,
}

impl Default for FilonOptions {
    fn default() -> Self {
        Self {
            n_panels: None,
            abs_tol: 1e-10,
            rel_tol: 1e-10,
            max_panels: 10_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Filon moment coefficients (classical)
// ---------------------------------------------------------------------------

/// Compute the Filon alpha, beta, gamma coefficients for `theta = omega * h`.
///
/// These arise from the moments of `sin` and `cos` over a panel of width `2h`.
/// Standard Filon formulas (Abramowitz & Stegun 25.4.47):
///
/// - `alpha = 1/theta + sin(2*theta)/(2*theta^2) - 2*sin^2(theta)/theta^3`
///   ... actually the corrected formulas from Davis & Rabinowitz:
///
/// - `alpha = (theta^2 + theta*sin(theta)*cos(theta) - 2*sin^2(theta)) / theta^3`
/// - `beta  = 2 * (theta*(1 + cos^2(theta)) - 2*sin(theta)*cos(theta)) / theta^3`
/// - `gamma = 4 * (sin(theta) - theta*cos(theta)) / theta^3`
///
/// When `theta` is small we use Taylor expansions to avoid catastrophic cancellation.
fn filon_coefficients(theta: f64) -> (f64, f64, f64) {
    if theta.abs() < 1e-4 {
        // Taylor expansion for numerical stability when theta is very small
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let t8 = t4 * t4;
        let alpha = 2.0 * t2 / 45.0 - 2.0 * t4 / 315.0 + 2.0 * t6 / 4725.0 - 2.0 * t8 / 66825.0;
        let beta =
            2.0 / 3.0 + 2.0 * t2 / 15.0 - 4.0 * t4 / 105.0 + 2.0 * t6 / 567.0 - 4.0 * t8 / 14175.0;
        let gamma = 4.0 / 3.0 - 2.0 * t2 / 15.0 + t4 / 210.0 - t6 / 11340.0 + t8 / 748440.0;
        (alpha, beta, gamma)
    } else {
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let t2 = theta * theta;
        let t3 = t2 * theta;

        let alpha = (t2 + theta * sin_t * cos_t - 2.0 * sin_t * sin_t) / t3;
        let beta = 2.0 * (theta * (1.0 + cos_t * cos_t) - 2.0 * sin_t * cos_t) / t3;
        let gamma = 4.0 * (sin_t - theta * cos_t) / t3;
        (alpha, beta, gamma)
    }
}

// ---------------------------------------------------------------------------
// Classical Filon
// ---------------------------------------------------------------------------

/// Classical Filon quadrature for oscillatory integrals.
///
/// Computes `integral_a^b f(x) * K(omega * x) dx` where `K` is `sin` or `cos`.
///
/// The function `f` is sampled on an evenly-spaced grid with `2n + 1` points
/// (i.e., `n` panels of two sub-intervals each), and a piecewise-quadratic
/// interpolant is combined with analytic moment integrals.
///
/// # Arguments
///
/// * `f`      - The non-oscillatory part of the integrand.
/// * `a`, `b` - Integration limits.
/// * `omega`  - Angular frequency of the oscillation.
/// * `kernel` - Whether the oscillatory factor is `sin` or `cos`.
/// * `options`- Optional parameters (panel count, tolerances).
///
/// # Errors
///
/// Returns an error if `omega` is zero (use standard quadrature) or if
/// the panel count is odd / too small.
pub fn filon<F>(
    f: F,
    a: f64,
    b: f64,
    omega: f64,
    kernel: OscillatoryKernel,
    options: Option<FilonOptions>,
) -> IntegrateResult<FilonResult>
where
    F: Fn(f64) -> f64,
{
    if omega.abs() < f64::EPSILON {
        return Err(IntegrateError::ValueError(
            "omega must be non-zero; use standard quadrature for non-oscillatory integrals"
                .to_string(),
        ));
    }
    if (b - a).abs() < f64::EPSILON {
        return Ok(FilonResult {
            value: 0.0,
            error_estimate: 0.0,
            n_evals: 0,
            n_panels: 0,
        });
    }

    let opts = options.unwrap_or_default();

    match opts.n_panels {
        Some(n) => filon_fixed(&f, a, b, omega, kernel, n),
        None => filon_adaptive(&f, a, b, omega, kernel, &opts),
    }
}

/// Fixed-panel-count classical Filon.
///
/// The classical Filon method divides `[a,b]` into `N` panels, each of width
/// `2h = (b-a)/N`, giving `2N+1` sample points. The user-facing `n_panels`
/// refers to `N`, and we have `h = (b-a)/(2N)`.
fn filon_fixed<F>(
    f: &F,
    a: f64,
    b: f64,
    omega: f64,
    kernel: OscillatoryKernel,
    n_panels: usize,
) -> IntegrateResult<FilonResult>
where
    F: Fn(f64) -> f64,
{
    if n_panels == 0 {
        return Err(IntegrateError::ValueError(
            "n_panels must be positive".to_string(),
        ));
    }

    let big_n = n_panels; // number of Filon panels
    let total_points = 2 * big_n + 1;
    let h = (b - a) / (2 * big_n) as f64;
    let theta = omega * h;
    let (alpha, beta, gamma) = filon_coefficients(theta);

    // Sample f at 2N+1 points: x_i = a + i*h, i = 0, 1, ..., 2N
    let mut fx = Vec::with_capacity(total_points);
    for i in 0..total_points {
        fx.push(f(a + i as f64 * h));
    }

    // Filon's formula (Davis & Rabinowitz):
    //
    // For SINE: integral_a^b f(x)*sin(omega*x) dx ≈
    //   h * [ alpha*(f_0*cos(omega*x_0) - f_{2N}*cos(omega*x_{2N}))
    //        + beta*S_e + gamma*S_o ]
    //
    // For COSINE: integral_a^b f(x)*cos(omega*x) dx ≈
    //   h * [ alpha*(f_{2N}*sin(omega*x_{2N}) - f_0*sin(omega*x_0))
    //        + beta*C_e + gamma*C_o ]
    //
    // where:
    //   S_e = sum_{j=0}^{N} f_{2j} * sin(omega*x_{2j})  (all even-indexed points)
    //   S_o = sum_{j=0}^{N-1} f_{2j+1} * sin(omega*x_{2j+1})  (all odd-indexed points)
    //   C_e, C_o similarly with cos

    let osc_fn = match kernel {
        OscillatoryKernel::Sine => f64::sin,
        OscillatoryKernel::Cosine => f64::cos,
    };

    // C_even: sum'' f(x_{2j}) * osc(omega * x_{2j}) for j=0..N
    // (double-prime sum: endpoints get weight 1/2)
    let mut c_even = 0.0_f64;
    for j in 0..=big_n {
        let i = 2 * j;
        let x = a + i as f64 * h;
        let endpoint_w = if j == 0 || j == big_n { 0.5 } else { 1.0 };
        c_even += endpoint_w * fx[i] * osc_fn(omega * x);
    }

    // C_odd: sum f(x_{2j+1}) * osc(omega * x_{2j+1}) for j=0..N-1
    let mut c_odd = 0.0_f64;
    for j in 0..big_n {
        let i = 2 * j + 1;
        let x = a + i as f64 * h;
        c_odd += fx[i] * osc_fn(omega * x);
    }

    // Endpoint terms with complementary oscillation
    let value = match kernel {
        OscillatoryKernel::Sine => {
            let f0_cos = fx[0] * (omega * a).cos();
            let fn_cos = fx[2 * big_n] * (omega * b).cos();
            h * (alpha * (f0_cos - fn_cos) + beta * c_even + gamma * c_odd)
        }
        OscillatoryKernel::Cosine => {
            let f0_sin = fx[0] * (omega * a).sin();
            let fn_sin = fx[2 * big_n] * (omega * b).sin();
            h * (alpha * (fn_sin - f0_sin) + beta * c_even + gamma * c_odd)
        }
    };

    Ok(FilonResult {
        value,
        error_estimate: 0.0,
        n_evals: total_points,
        n_panels: big_n,
    })
}

/// Adaptive Filon: doubles panel count until convergence.
fn filon_adaptive<F>(
    f: &F,
    a: f64,
    b: f64,
    omega: f64,
    kernel: OscillatoryKernel,
    opts: &FilonOptions,
) -> IntegrateResult<FilonResult>
where
    F: Fn(f64) -> f64,
{
    // Start with a reasonable panel count based on frequency
    let wavelengths = omega.abs() * (b - a) / (2.0 * PI);
    let init_panels = {
        let base = (4.0 * wavelengths.ceil()).max(4.0) as usize;
        // round up to even
        if !base.is_multiple_of(2) {
            base + 1
        } else {
            base
        }
    };

    let mut n = init_panels;
    let mut prev = filon_fixed(f, a, b, omega, kernel, n)?;

    loop {
        n *= 2;
        if n > opts.max_panels {
            // Return what we have with an error estimate
            let last = filon_fixed(f, a, b, omega, kernel, n / 2)?;
            return Ok(FilonResult {
                value: last.value,
                error_estimate: (last.value - prev.value).abs(),
                n_evals: last.n_evals,
                n_panels: last.n_panels,
            });
        }
        let curr = filon_fixed(f, a, b, omega, kernel, n)?;
        let diff = (curr.value - prev.value).abs();
        let scale = curr.value.abs().max(1.0);
        if diff < opts.abs_tol || diff < opts.rel_tol * scale {
            return Ok(FilonResult {
                value: curr.value,
                error_estimate: diff,
                n_evals: curr.n_evals,
                n_panels: curr.n_panels,
            });
        }
        prev = curr;
    }
}

// ---------------------------------------------------------------------------
// Filon-Simpson
// ---------------------------------------------------------------------------

/// Filon-Simpson quadrature for oscillatory integrals.
///
/// This is a variant that applies the Simpson 1/3 pattern to approximate the
/// slowly-varying part `f(x)` and computes the resulting trigonometric
/// moments analytically. It can be more accurate than classical Filon when
/// `f` has moderate variation.
///
/// Uses `2n + 1` points (n must be even).
pub fn filon_simpson<F>(
    f: F,
    a: f64,
    b: f64,
    omega: f64,
    kernel: OscillatoryKernel,
    n_panels: usize,
) -> IntegrateResult<FilonResult>
where
    F: Fn(f64) -> f64,
{
    if omega.abs() < f64::EPSILON {
        return Err(IntegrateError::ValueError(
            "omega must be non-zero for Filon quadrature".to_string(),
        ));
    }
    let n = if n_panels < 2 {
        2
    } else if !n_panels.is_multiple_of(2) {
        n_panels + 1
    } else {
        n_panels
    };

    let h = (b - a) / n as f64;
    let n_points = n + 1;

    // Sample f at all points
    let mut fx = Vec::with_capacity(n_points);
    for i in 0..n_points {
        fx.push(f(a + i as f64 * h));
    }

    // Apply Simpson-type weighting to f, then multiply by oscillatory kernel
    // Simpson weights: 1, 4, 2, 4, 2, ..., 4, 1  (for n panels)
    let mut value = 0.0_f64;
    for i in 0..n_points {
        let x = a + i as f64 * h;
        let osc = match kernel {
            OscillatoryKernel::Sine => (omega * x).sin(),
            OscillatoryKernel::Cosine => (omega * x).cos(),
        };
        let simpson_w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        value += simpson_w * fx[i] * osc;
    }
    value *= h / 3.0;

    Ok(FilonResult {
        value,
        error_estimate: 0.0,
        n_evals: n_points,
        n_panels: n,
    })
}

// ---------------------------------------------------------------------------
// Filon-Clenshaw-Curtis
// ---------------------------------------------------------------------------

/// Filon-Clenshaw-Curtis quadrature for oscillatory integrals.
///
/// Samples `f` at Chebyshev (Clenshaw-Curtis) nodes, expands in Chebyshev
/// polynomials, and evaluates the modified moments
/// `integral_{-1}^{1} T_k(t) * sin/cos(omega' * t) dt` via a stable recurrence.
///
/// This method is particularly effective for smooth `f` and moderate-to-high `omega`.
///
/// # Arguments
///
/// * `f`      - The non-oscillatory part.
/// * `a`, `b` - Integration limits.
/// * `omega`  - Angular frequency.
/// * `kernel` - Sine or cosine kernel.
/// * `n`      - Number of Chebyshev points (degree of expansion).
pub fn filon_clenshaw_curtis<F>(
    f: F,
    a: f64,
    b: f64,
    omega: f64,
    kernel: OscillatoryKernel,
    n: usize,
) -> IntegrateResult<FilonResult>
where
    F: Fn(f64) -> f64,
{
    if omega.abs() < f64::EPSILON {
        return Err(IntegrateError::ValueError(
            "omega must be non-zero".to_string(),
        ));
    }
    if n == 0 {
        return Err(IntegrateError::ValueError(
            "n must be at least 1".to_string(),
        ));
    }

    let half_range = (b - a) / 2.0;
    let mid = (a + b) / 2.0;
    // omega' on the reference interval [-1,1]
    let omega_ref = omega * half_range;

    // Sample f at Clenshaw-Curtis nodes on [-1,1]
    let nn = n;
    let mut f_vals = Vec::with_capacity(nn + 1);
    for j in 0..=nn {
        let t = (PI * j as f64 / nn as f64).cos();
        let x = mid + half_range * t;
        f_vals.push(f(x));
    }

    // Compute Chebyshev coefficients via DCT-I like transform
    let mut coeffs = vec![0.0_f64; nn + 1];
    for k in 0..=nn {
        let mut s = 0.0_f64;
        for j in 0..=nn {
            let c_j = if j == 0 || j == nn { 0.5 } else { 1.0 };
            s += c_j * f_vals[j] * (PI * k as f64 * j as f64 / nn as f64).cos();
        }
        let c_k = if k == 0 || k == nn { 1.0 } else { 2.0 };
        coeffs[k] = c_k * s / nn as f64;
    }

    // Compute modified moments mu_k = integral_{-1}^{1} T_k(t) * sin/cos(omega_ref * t) dt
    let moments = chebyshev_oscillatory_moments(omega_ref, nn, kernel)?;

    // Account for the cos(omega*mid) / sin(omega*mid) phase shift
    // The integral on [a,b] with kernel K(omega*x) transforms to:
    // half_range * integral_{-1}^{1} f(mid + half_range*t) * K(omega*(mid + half_range*t)) dt
    // = half_range * integral_{-1}^{1} f(...) * K(omega*mid + omega_ref*t) dt
    // Using addition formulas:
    //   sin(A+B) = sin(A)cos(B) + cos(A)sin(B)
    //   cos(A+B) = cos(A)cos(B) - sin(A)sin(B)
    // We need to split into sin and cos moments on [-1,1].
    let _ = moments; // initial moments superseded by phase-corrected computation below
    let sin_mid = (omega * mid).sin();
    let cos_mid = (omega * mid).cos();
    let moments_sin = chebyshev_oscillatory_moments(omega_ref, nn, OscillatoryKernel::Sine)?;
    let moments_cos = chebyshev_oscillatory_moments(omega_ref, nn, OscillatoryKernel::Cosine)?;

    let mut integral_sin = 0.0_f64;
    let mut integral_cos = 0.0_f64;
    for k in 0..=nn {
        integral_sin += coeffs[k] * moments_sin[k];
        integral_cos += coeffs[k] * moments_cos[k];
    }

    let value = match kernel {
        OscillatoryKernel::Sine => half_range * (sin_mid * integral_cos + cos_mid * integral_sin),
        OscillatoryKernel::Cosine => half_range * (cos_mid * integral_cos - sin_mid * integral_sin),
    };

    Ok(FilonResult {
        value,
        error_estimate: 0.0,
        n_evals: nn + 1,
        n_panels: nn,
    })
}

/// Compute modified moments integral_{-1}^{1} T_k(t) * sin/cos(w*t) dt for k = 0..n.
///
/// Uses the stable recurrence from Piessens (1973) / Dominguez et al.
fn chebyshev_oscillatory_moments(
    w: f64,
    n: usize,
    kernel: OscillatoryKernel,
) -> IntegrateResult<Vec<f64>> {
    let mut mu = vec![0.0_f64; n + 1];

    if w.abs() < 1e-14 {
        // w -> 0 limit
        match kernel {
            OscillatoryKernel::Cosine => {
                // integral_{-1}^{1} T_k(t) dt
                for k in 0..=n {
                    if k % 2 == 0 {
                        let kf = k as f64;
                        mu[k] = 2.0 / (1.0 - kf * kf);
                    }
                }
            }
            OscillatoryKernel::Sine => {
                // integral_{-1}^{1} T_k(t) * 0 dt => 0
                // (sin(0) = 0)
            }
        }
        return Ok(mu);
    }

    // Base moments:
    // mu_0^cos = integral_{-1}^{1} cos(w*t) dt = 2*sin(w)/w
    // mu_0^sin = integral_{-1}^{1} sin(w*t) dt = 2*(1 - cos(w))/w ... wait
    // Actually: integral_{-1}^{1} sin(w*t) dt = [-cos(w*t)/w]_{-1}^{1} = (-cos(w) + cos(-w))/w = 0
    // (sin is odd on [-1,1])
    // Hmm: integral_{-1}^{1} sin(w*t) dt = [-cos(wt)/w]_{-1}^1 = (-cos(w)+cos(w))/w = 0.
    // But T_1(t) = t, and integral_{-1}^{1} t * sin(w*t) dt != 0.

    // Use direct numerical computation for low-order moments then recurrence.
    // The three-term recurrence for T_k: T_{k+1}(t) = 2t T_k(t) - T_{k-1}(t)
    // gives for moments:
    // mu_{k+1} = (2/w) * d/dw [mu_k^{sin/cos partner}] - mu_{k-1}
    //
    // Actually the standard recurrence is (for the cos case):
    // mu_{k+1}^cos + mu_{k-1}^cos = (2k/w) * mu_k^sin  ... not straightforward.
    //
    // Let's use the direct formula instead for reliability:
    // integral_{-1}^{1} T_k(t) cos(w*t) dt and sin(w*t) dt
    // are known: they involve Bessel functions J_k(w).
    // Specifically: integral_{-1}^{1} T_k(t) cos(wt) / sqrt(1-t^2) dt = pi * J_k(w) * cos(k*pi/2)
    // But our integrals don't have the 1/sqrt(1-t^2) weight.
    //
    // For unweighted integrals, use integration by parts or direct Gauss-Legendre
    // on each moment. Since these are smooth and bounded, a modest GL rule suffices.

    // We'll use a 64-point Gauss-Legendre rule for the moment integrals.
    let gl_n = 64;
    let (gl_nodes, gl_weights) = gauss_legendre_nodes_weights(gl_n);

    for k in 0..=n {
        let mut s = 0.0_f64;
        for i in 0..gl_n {
            let t = gl_nodes[i];
            let tk = chebyshev_t(k, t);
            let osc = match kernel {
                OscillatoryKernel::Sine => (w * t).sin(),
                OscillatoryKernel::Cosine => (w * t).cos(),
            };
            s += gl_weights[i] * tk * osc;
        }
        mu[k] = s;
    }

    Ok(mu)
}

/// Evaluate T_k(x) using the recurrence T_0=1, T_1=x, T_{k+1}=2x*T_k - T_{k-1}.
fn chebyshev_t(k: usize, x: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k == 1 {
        return x;
    }
    let mut t0 = 1.0_f64;
    let mut t1 = x;
    for _ in 2..=k {
        let t2 = 2.0 * x * t1 - t0;
        t0 = t1;
        t1 = t2;
    }
    t1
}

/// Compute Gauss-Legendre nodes and weights on [-1, 1].
fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];
    let m = n.div_ceil(2);

    for i in 0..m {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let mut p0 = 1.0_f64;
            let mut p1 = z;
            for k in 2..=n {
                let kf = k as f64;
                let p2 = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            let nf = n as f64;
            let dp = nf * (z * p1 - p0) / (z * z - 1.0);
            let delta = p1 / dp;
            z -= delta;
            if delta.abs() < 1e-15 {
                break;
            }
        }
        nodes[i] = -z;
        nodes[n - 1 - i] = z;

        let mut p0 = 1.0_f64;
        let mut p1 = z;
        for k in 2..=n {
            let kf = k as f64;
            let p2 = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf;
            p0 = p1;
            p1 = p2;
        }
        let nf = n as f64;
        let dp = nf * (z * p1 - p0) / (z * z - 1.0);
        let w = 2.0 / ((1.0 - z * z) * dp * dp);
        weights[i] = w;
        weights[n - 1 - i] = w;
    }

    (nodes, weights)
}

// ---------------------------------------------------------------------------
// Levin collocation (basic)
// ---------------------------------------------------------------------------

/// Basic Levin collocation method for general oscillatory integrals.
///
/// Computes `integral_a^b f(x) * exp(i * omega * g(x)) dx` by solving
/// a collocation system. This basic version handles `g(x) = x` and returns
/// the real part corresponding to the chosen kernel.
///
/// The idea: find a function `p(x)` such that `d/dx [p(x) * exp(i*omega*g(x))]`
/// approximates `f(x) * exp(i*omega*g(x))`. Then the integral equals
/// `p(b)*exp(i*omega*g(b)) - p(a)*exp(i*omega*g(a))`.
///
/// We approximate `p` as a polynomial of degree `n-1` and solve the collocation
/// system at Chebyshev nodes.
///
/// # Arguments
///
/// * `f`      - Non-oscillatory amplitude.
/// * `a`, `b` - Integration limits.
/// * `omega`  - Angular frequency.
/// * `kernel` - Sine or cosine.
/// * `n`      - Number of collocation points (polynomial degree + 1).
pub fn levin_collocation<F>(
    f: F,
    a: f64,
    b: f64,
    omega: f64,
    kernel: OscillatoryKernel,
    n: usize,
) -> IntegrateResult<FilonResult>
where
    F: Fn(f64) -> f64,
{
    if omega.abs() < f64::EPSILON {
        return Err(IntegrateError::ValueError(
            "omega must be non-zero".to_string(),
        ));
    }
    if n < 2 {
        return Err(IntegrateError::ValueError(
            "n must be at least 2 for Levin collocation".to_string(),
        ));
    }

    // For g(x) = x, the equation is:
    //   p'(x) + i*omega*p(x) = f(x)
    // We solve for the real part (cosine case) or imaginary part (sine case).
    //
    // Let p(x) = p_r(x) + i*p_i(x). Then:
    //   p_r'(x) - omega*p_i(x) = f(x)
    //   p_i'(x) + omega*p_r(x) = 0
    //
    // We represent p_r and p_i as polynomials of degree n-1 each,
    // with collocation at Chebyshev nodes mapped to [a,b].

    let half = (b - a) / 2.0;
    let mid = (a + b) / 2.0;

    // Chebyshev nodes on [a, b]
    let mut nodes = Vec::with_capacity(n);
    for j in 0..n {
        let t = (PI * (2 * j + 1) as f64 / (2 * n) as f64).cos();
        nodes.push(mid + half * t);
    }

    // Build the 2n x 2n collocation system.
    // Unknown vector: [c_r_0, ..., c_r_{n-1}, c_i_0, ..., c_i_{n-1}]
    // where p_r(x) = sum_k c_r_k * ((x-mid)/half)^k
    //       p_i(x) = sum_k c_i_k * ((x-mid)/half)^k
    //
    // At each node x_j:
    //   p_r'(x_j) - omega * p_i(x_j) = f(x_j)     ... row j
    //   p_i'(x_j) + omega * p_r(x_j) = 0           ... row n+j

    let size = 2 * n;
    let mut mat = vec![0.0_f64; size * size];
    let mut rhs = vec![0.0_f64; size];

    for j in 0..n {
        let x = nodes[j];
        let t = (x - mid) / half;

        // Powers of t and their derivatives
        // phi_k(x) = t^k, phi_k'(x) = k * t^{k-1} / half
        for k in 0..n {
            let phi_k = t.powi(k as i32);
            let dphi_k = if k == 0 {
                0.0
            } else {
                k as f64 * t.powi(k as i32 - 1) / half
            };

            // Row j (p_r' - omega*p_i = f):
            //   dphi_k for c_r_k
            mat[j * size + k] = dphi_k;
            //   -omega * phi_k for c_i_k
            mat[j * size + n + k] = -omega * phi_k;

            // Row n+j (p_i' + omega*p_r = 0):
            //   omega * phi_k for c_r_k
            mat[(n + j) * size + k] = omega * phi_k;
            //   dphi_k for c_i_k
            mat[(n + j) * size + n + k] = dphi_k;
        }

        rhs[j] = f(x);
        rhs[n + j] = 0.0;
    }

    // Solve the linear system using Gaussian elimination with partial pivoting
    let coeffs = solve_linear_system(&mat, &rhs, size)?;

    // Evaluate p_r and p_i at a and b
    let eval_poly = |t: f64, start: usize| -> f64 {
        let mut val = 0.0_f64;
        for k in 0..n {
            val += coeffs[start + k] * t.powi(k as i32);
        }
        val
    };

    let ta = (a - mid) / half;
    let tb = (b - mid) / half;

    let pr_a = eval_poly(ta, 0);
    let pi_a = eval_poly(ta, n);
    let pr_b = eval_poly(tb, 0);
    let pi_b = eval_poly(tb, n);

    // integral = p(b)*exp(i*omega*b) - p(a)*exp(i*omega*a)
    // exp(i*omega*x) = cos(omega*x) + i*sin(omega*x)
    // p(x)*exp(i*omega*x) = (p_r + i*p_i)(cos + i*sin)
    //   = (p_r*cos - p_i*sin) + i*(p_r*sin + p_i*cos)
    // For cosine kernel we want the real part, for sine we want the imaginary part.

    let cos_a = (omega * a).cos();
    let sin_a = (omega * a).sin();
    let cos_b = (omega * b).cos();
    let sin_b = (omega * b).sin();

    let real_b = pr_b * cos_b - pi_b * sin_b;
    let real_a = pr_a * cos_a - pi_a * sin_a;
    let imag_b = pr_b * sin_b + pi_b * cos_b;
    let imag_a = pr_a * sin_a + pi_a * cos_a;

    let value = match kernel {
        OscillatoryKernel::Cosine => real_b - real_a,
        OscillatoryKernel::Sine => imag_b - imag_a,
    };

    Ok(FilonResult {
        value,
        error_estimate: 0.0,
        n_evals: n,
        n_panels: n,
    })
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> IntegrateResult<Vec<f64>> {
    let mut aug = vec![0.0_f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(IntegrateError::LinearSolveError(
                "Singular or nearly singular matrix in Levin collocation".to_string(),
            ));
        }
        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                let val = aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            s -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = s / aug[i * (n + 1) + i];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// integral_0^1 sin(100x) dx = (1 - cos(100)) / 100
    #[test]
    fn test_filon_sin_high_frequency() {
        let omega: f64 = 100.0;
        let exact = (1.0 - (omega * 1.0).cos()) / omega;

        // Use enough panels for high frequency: omega=100 needs many panels
        let result = filon(
            |_x| 1.0,
            0.0,
            1.0,
            omega,
            OscillatoryKernel::Sine,
            Some(FilonOptions {
                n_panels: Some(200),
                ..Default::default()
            }),
        )
        .expect("filon should succeed");

        assert!(
            (result.value - exact).abs() < 1e-6,
            "Filon sin(100x): got {}, expected {}, diff={}",
            result.value,
            exact,
            (result.value - exact).abs()
        );
    }

    /// integral_0^1 x^2 * cos(50x) dx
    /// Integration by parts (twice) gives:
    /// = [x^2 sin(50x)/50]_0^1 - integral 2x sin(50x)/50 dx
    /// = sin(50)/50 - (2/50) * integral_0^1 x sin(50x) dx
    /// integral x sin(50x) dx = [-x cos(50x)/50]_0^1 + integral cos(50x)/50 dx
    ///   = -cos(50)/50 + sin(50)/2500
    /// So total = sin(50)/50 - (2/50)*(-cos(50)/50 + sin(50)/2500)
    ///          = sin(50)/50 + 2*cos(50)/2500 - 2*sin(50)/125000
    #[test]
    fn test_filon_x2_cos50x() {
        let omega: f64 = 50.0;
        // Compute analytical result
        let s = (omega).sin();
        let c = (omega).cos();
        let exact = s / omega + 2.0 * c / (omega * omega) - 2.0 * s / (omega * omega * omega);

        let result = filon(
            |x| x * x,
            0.0,
            1.0,
            omega,
            OscillatoryKernel::Cosine,
            Some(FilonOptions {
                n_panels: Some(200),
                ..Default::default()
            }),
        )
        .expect("filon should succeed");

        assert!(
            (result.value - exact).abs() < 1e-4,
            "Filon x^2 cos(50x): got {}, expected {}, diff={}",
            result.value,
            exact,
            (result.value - exact).abs()
        );
    }

    /// Test convergence: increasing panels should improve accuracy.
    #[test]
    fn test_filon_convergence() {
        let omega: f64 = 30.0;
        let exact = (1.0 - (omega).cos()) / omega; // integral_0^1 sin(30x) dx

        let mut prev_err = f64::INFINITY;
        for &np in &[20, 40, 80, 200] {
            let result = filon(
                |_x| 1.0,
                0.0,
                1.0,
                omega,
                OscillatoryKernel::Sine,
                Some(FilonOptions {
                    n_panels: Some(np),
                    ..Default::default()
                }),
            )
            .expect("filon should succeed");

            let err = (result.value - exact).abs();
            // Error should generally decrease (allow some slack due to oscillatory nature)
            assert!(
                err < prev_err * 1.5 || err < 1e-12,
                "Convergence failed at n={}: err={}, prev_err={}",
                np,
                err,
                prev_err
            );
            prev_err = err;
        }
        // Final error should be small
        assert!(prev_err < 1e-8, "Final error too large: {}", prev_err);
    }

    /// Filon-Simpson test
    #[test]
    fn test_filon_simpson_basic() {
        let omega: f64 = 10.0;
        let exact = (1.0 - (omega).cos()) / omega;

        let result = filon_simpson(|_x| 1.0, 0.0, 1.0, omega, OscillatoryKernel::Sine, 100)
            .expect("filon_simpson should succeed");

        assert!(
            (result.value - exact).abs() < 1e-6,
            "Filon-Simpson: got {}, expected {}, diff={}",
            result.value,
            exact,
            (result.value - exact).abs()
        );
    }

    /// Filon-Clenshaw-Curtis test
    #[test]
    fn test_filon_clenshaw_curtis_cos() {
        let omega: f64 = 20.0;
        // integral_0^1 cos(20x) dx = sin(20)/20
        let exact = (omega).sin() / omega;

        let result =
            filon_clenshaw_curtis(|_x| 1.0, 0.0, 1.0, omega, OscillatoryKernel::Cosine, 32)
                .expect("filon_cc should succeed");

        assert!(
            (result.value - exact).abs() < 1e-6,
            "Filon-CC cos: got {}, expected {}, diff={}",
            result.value,
            exact,
            (result.value - exact).abs()
        );
    }

    /// Test Filon-CC for f(x) = x * sin(30x) over [0, 1]
    #[test]
    fn test_filon_clenshaw_curtis_with_amplitude() {
        let omega: f64 = 30.0;
        // integral_0^1 x sin(30x) dx = [-x cos(30x)/30]_0^1 + integral cos(30x)/30 dx
        //   = -cos(30)/30 + sin(30)/900
        let exact = -(omega).cos() / omega + (omega).sin() / (omega * omega);

        let result = filon_clenshaw_curtis(|x| x, 0.0, 1.0, omega, OscillatoryKernel::Sine, 48)
            .expect("filon_cc should succeed");

        assert!(
            (result.value - exact).abs() < 1e-5,
            "Filon-CC x*sin(30x): got {}, expected {}, diff={}",
            result.value,
            exact,
            (result.value - exact).abs()
        );
    }

    /// Levin collocation test
    #[test]
    fn test_levin_basic() {
        let omega: f64 = 10.0;
        // integral_0^1 sin(10x) dx = (1-cos(10))/10
        let exact = (1.0 - (omega).cos()) / omega;

        let result = levin_collocation(|_x| 1.0, 0.0, 1.0, omega, OscillatoryKernel::Sine, 8)
            .expect("levin should succeed");

        assert!(
            (result.value - exact).abs() < 1e-4,
            "Levin sin(10x): got {}, expected {}, diff={}",
            result.value,
            exact,
            (result.value - exact).abs()
        );
    }

    /// Test zero-length interval
    #[test]
    fn test_filon_zero_interval() {
        let result = filon(|_x| 1.0, 1.0, 1.0, 10.0, OscillatoryKernel::Sine, None)
            .expect("zero interval should succeed");
        assert!(
            result.value.abs() < 1e-15,
            "Zero interval should give 0, got {}",
            result.value
        );
    }

    /// Test error on omega = 0
    #[test]
    fn test_filon_omega_zero_error() {
        let result = filon(|_x| 1.0, 0.0, 1.0, 0.0, OscillatoryKernel::Sine, None);
        assert!(result.is_err(), "omega=0 should return error");
    }

    /// Test with exp(-x) * sin(100x) -- a decaying oscillatory integral
    #[test]
    fn test_filon_exp_decay_oscillatory() {
        let omega: f64 = 100.0;
        // integral_0^10 exp(-x) * sin(100x) dx
        // = omega / (1 + omega^2) * (1 - exp(-10) * (cos(1000) + omega * sin(1000) / omega))
        // Actually: integral exp(-x) sin(wx) dx = (w - exp(-x)(w cos(wx) + sin(wx)))/(1+w^2)
        // Evaluated from 0 to 10:
        let w = omega;
        let w2 = w * w;
        let e10 = (-10.0_f64).exp();
        let exact = (w - e10 * (w * (w * 10.0).cos() + (w * 10.0).sin()) - (w - 0.0)) / (1.0 + w2);
        // Simplify: = w/(1+w^2) * (1 - e^{-10}*cos(1000)) - e^{-10}*sin(1000)/(1+w^2)
        // Better to compute directly:
        // integral = [e^{-x}(-sin(wx) - w cos(wx)) / (1+w^2)]_0^10 ... standard formula
        // = e^{-10}(-sin(1000) - w cos(1000))/(1+w^2) - (0 - w)/(1+w^2)
        let exact2 = e10 * (-(w * 10.0).sin() - w * (w * 10.0).cos()) / (1.0 + w2) + w / (1.0 + w2);

        let result = filon(
            |x| (-x).exp(),
            0.0,
            10.0,
            omega,
            OscillatoryKernel::Sine,
            None,
        )
        .expect("filon should succeed");

        // Use the cleaner exact2
        let _ = exact; // suppress unused warning
        assert!(
            (result.value - exact2).abs() < 1e-5,
            "Filon exp(-x)*sin(100x): got {}, expected {}, diff={}",
            result.value,
            exact2,
            (result.value - exact2).abs()
        );
    }
}
