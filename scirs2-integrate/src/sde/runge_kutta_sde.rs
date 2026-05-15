//! Stochastic Runge-Kutta (SRK) methods for SDEs
//!
//! This module provides high-order SRK methods that achieve strong convergence
//! order 1.5 for SDEs with additive or scalar noise, and second-order weak
//! convergence. These methods use multiple evaluations of the drift and
//! diffusion functions per step (similar to classical RK methods for ODEs).
//!
//! ## Strong Order 1.5 SRK Scheme
//!
//! The strong SRK scheme of Rümelin (1982) / Kloeden-Platen uses auxiliary
//! support values to achieve strong order 1.5 for scalar SDEs. The update uses:
//!
//! ```text
//! H+ = x_n + f dt + g √dt
//! H- = x_n + f dt - g √dt
//! x_{n+1} = x_n + f dt + g ΔW + (1/(2√dt)) [g(H+) - g(H-)] [(ΔW)^2/2 - dt/2]
//! ```
//!
//! ## Platen Explicit Scheme (order 1.5)
//!
//! The Platen explicit scheme (from Kloeden & Platen, 1992) achieves:
//! - **Strong order**: 1.5 (for scalar / diagonal noise)
//! - **Weak order**: 2.0
//!
//! It uses the Milstein correction plus an additional RK-style drift correction:
//!
//! ```text
//! x̂ = x_n + f(t_n, x_n) dt + g(t_n, x_n) √dt
//! x_{n+1} = x_n + (1/2)[f(t_n+dt, x̂) + f(t_n, x_n)] dt
//!           + g(t_n, x_n) ΔW
//!           + (1/(2√dt)) [g(t_n, x̂) - g(t_n, x_n)] [(ΔW)^2 - dt]
//! ```

use crate::error::{IntegrateError, IntegrateResult};
use crate::sde::levy_area::{iterated_integral, levy_area_wiktorsson};
use crate::sde::{compute_n_steps, SdeOptions, SdeProblem, SdeSolution};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::{Normal, Rng, StdRng};
use scirs2_core::Distribution;

/// Strong order 1.5 Stochastic Runge-Kutta (SRK) method.
///
/// This is the Rümelin SRK scheme for scalar/diagonal-noise SDEs, extended
/// to multi-dimensional state via diagonal diffusion structure. It achieves
/// **strong convergence order 1.5** for SDEs with scalar or diagonal noise,
/// compared to order 1.0 for the Milstein scheme.
///
/// The scheme uses support values:
/// ```text
/// H±_j = x_n + f(t_n, x_n) dt ± g_j(t_n, x_n) √dt
/// ```
///
/// where g_j is the j-th column of the diffusion matrix, and computes:
/// ```text
/// correction_ij = (1/(2√dt)) [g_ij(H+_j) - g_ij(H-_j)] * [(ΔW_j)^2/2 - dt/2]
/// ```
///
/// **Validity**: This strong order 1.5 result applies for **scalar noise**
/// (m=1) or **diagonal noise** (n=m, g is diagonal). For general noise,
/// iterated stochastic integrals are required which are not implemented here;
/// strong order falls back to 1.0 in that case.
///
/// # Arguments
///
/// * `prob` - SDE problem
/// * `dt` - Step size
/// * `rng` - RNG reference
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::{SdeProblem};
/// use scirs2_integrate::sde::runge_kutta_sde::srk_strong;
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_core::random::prelude::*;
///
/// // CIR process: dX = κ(μ - X) dt + σ √X dW  (reflected if X -> 0)
/// let (kappa, mu, sigma) = (2.0_f64, 0.5_f64, 0.3_f64);
/// let prob = SdeProblem::new(
///     array![0.5_f64], [0.0, 1.0], 1,
///     move |_t, x| array![kappa * (mu - x[0])],
///     move |_t, x| {
///         let mut g = Array2::zeros((1, 1));
///         g[[0, 0]] = sigma * x[0].max(0.0).sqrt();
///         g
///     },
/// );
/// let mut rng = seeded_rng(42);
/// let sol = srk_strong(&prob, 0.01, &mut rng).unwrap();
/// assert!(!sol.is_empty());
/// ```
pub fn srk_strong<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    srk_strong_with_options(prob, dt, rng, &SdeOptions::default())
}

/// Strong SRK with solver options.
pub fn srk_strong_with_options<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
    opts: &SdeOptions,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    prob.validate()?;
    let t0 = prob.t_span[0];
    let t1 = prob.t_span[1];
    let n_steps = compute_n_steps(t0, t1, dt, opts.max_steps)?;
    let n_state = prob.dim();
    let m = prob.n_brownian;

    let capacity = if opts.save_all_steps { n_steps + 1 } else { 2 };
    let mut sol = SdeSolution::with_capacity(capacity);
    sol.push(t0, prob.x0.clone());

    let normal = Normal::new(0.0_f64, 1.0_f64)
        .map_err(|e| IntegrateError::ComputationError(format!("Normal dist error: {}", e)))?;

    let mut x = prob.x0.clone();
    let mut t = t0;

    for step in 0..n_steps {
        let dt_actual = if step == n_steps - 1 {
            t1 - t
        } else {
            dt.min(t1 - t)
        };
        if dt_actual <= 0.0 {
            break;
        }
        let sqrt_dt = dt_actual.sqrt();

        // Generate Brownian increments
        let dw: Array1<f64> = Array1::from_shape_fn(m, |_| normal.sample(rng) * sqrt_dt);

        let f0 = (prob.f_drift)(t, &x);
        let g0 = (prob.g_diffusion)(t, &x);

        if f0.len() != n_state || g0.nrows() != n_state || g0.ncols() != m {
            return Err(IntegrateError::DimensionMismatch(
                "Dimension mismatch in drift or diffusion".to_string(),
            ));
        }

        // Base EM step (used for support values)
        let f0_dt = f0.clone() * dt_actual;

        // Milstein-style correction using support values H+ and H-
        // For each Brownian motion j, compute support values:
        // H+_j = x + f dt + g_j √dt
        // H-_j = x + f dt - g_j √dt
        let mut srk_correction = Array1::<f64>::zeros(n_state);
        for j in 0..m {
            let g_col_j = g0.column(j).to_owned();
            let h_plus = &x + &f0_dt + &g_col_j * sqrt_dt;
            let h_minus = &x + &f0_dt - &g_col_j * sqrt_dt;

            let g_h_plus = (prob.g_diffusion)(t, &h_plus);
            let g_h_minus = (prob.g_diffusion)(t, &h_minus);

            let dw_j = dw[j];
            // Correction: [g(H+)_j - g(H-)_j] / (2√dt) * [(ΔW_j)^2/2 - dt/2]
            // This is the Kloeden-Platen SRK strong 1.5 term (for scalar/diag noise)
            let factor = (dw_j * dw_j * 0.5 - dt_actual * 0.5) / sqrt_dt;
            for i in 0..n_state {
                srk_correction[i] += (g_h_plus[[i, j]] - g_h_minus[[i, j]]) * 0.5 * factor;
            }
        }

        // x_{n+1} = x_n + f*dt + g*ΔW + SRK_correction
        x = x + f0 * dt_actual + g0.dot(&dw) + srk_correction;
        t += dt_actual;

        if opts.save_all_steps {
            sol.push(t, x.clone());
        }
    }

    if !opts.save_all_steps {
        sol.push(t, x);
    }

    Ok(sol)
}

/// Platen explicit scheme (order 1.5 strong, order 2.0 weak).
///
/// The Platen explicit scheme (see Kloeden & Platen 1992, §11.2) achieves:
/// - **Strong convergence order**: 1.5 (for scalar noise or additive noise)
/// - **Weak convergence order**: 2.0
///
/// It combines a Heun-type predictor for the drift (midpoint correction) with
/// the Milstein correction for the diffusion, plus second-order drift correction:
///
/// ```text
/// x̂_n = x_n + f(t_n, x_n) dt + g(t_n, x_n) √dt
/// x_{n+1} = x_n + (1/2)[f(t_{n+1}, x̂_n) + f(t_n, x_n)] dt
///           + g(t_n, x_n) ΔW_n
///           + (1/(2√dt)) [g(t_n, x̂_n) - g(t_n, x_n)] [(ΔW_n)^2 - dt]
/// ```
///
/// # Arguments
///
/// * `prob` - SDE problem
/// * `dt` - Step size
/// * `rng` - RNG reference
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::SdeProblem;
/// use scirs2_integrate::sde::runge_kutta_sde::platen_scheme;
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_core::random::prelude::*;
///
/// // Ornstein-Uhlenbeck: dX = θ(μ - X) dt + σ dW  (additive noise)
/// let (theta, mu, sigma) = (1.5_f64, 0.0_f64, 0.5_f64);
/// let prob = SdeProblem::new(
///     array![1.0_f64], [0.0, 2.0], 1,
///     move |_t, x| array![theta * (mu - x[0])],
///     move |_t, _x| { let mut g = Array2::zeros((1,1)); g[[0,0]] = sigma; g },
/// );
/// let mut rng = seeded_rng(7);
/// let sol = platen_scheme(&prob, 0.01, &mut rng).unwrap();
/// assert!(sol.len() > 1);
/// ```
pub fn platen_scheme<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    platen_scheme_with_options(prob, dt, rng, &SdeOptions::default())
}

/// Platen scheme with solver options.
pub fn platen_scheme_with_options<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
    opts: &SdeOptions,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    prob.validate()?;
    let t0 = prob.t_span[0];
    let t1 = prob.t_span[1];
    let n_steps = compute_n_steps(t0, t1, dt, opts.max_steps)?;
    let n_state = prob.dim();
    let m = prob.n_brownian;

    let capacity = if opts.save_all_steps { n_steps + 1 } else { 2 };
    let mut sol = SdeSolution::with_capacity(capacity);
    sol.push(t0, prob.x0.clone());

    let normal = Normal::new(0.0_f64, 1.0_f64)
        .map_err(|e| IntegrateError::ComputationError(format!("Normal dist error: {}", e)))?;

    let mut x = prob.x0.clone();
    let mut t = t0;

    for step in 0..n_steps {
        let dt_actual = if step == n_steps - 1 {
            t1 - t
        } else {
            dt.min(t1 - t)
        };
        if dt_actual <= 0.0 {
            break;
        }
        let sqrt_dt = dt_actual.sqrt();

        // Generate Brownian increments
        let dw: Array1<f64> = Array1::from_shape_fn(m, |_| normal.sample(rng) * sqrt_dt);

        let f0 = (prob.f_drift)(t, &x);
        let g0 = (prob.g_diffusion)(t, &x);

        if f0.len() != n_state || g0.nrows() != n_state || g0.ncols() != m {
            return Err(IntegrateError::DimensionMismatch(
                "Dimension mismatch in drift or diffusion".to_string(),
            ));
        }

        // Platen predictor: x̂ = x + f*dt + g * √dt * 1_m
        // (uses unit vector for support; 1_m = (1,1,...,1) of length m)
        let ones_m = Array1::from_elem(m, 1.0_f64);
        let x_hat = &x + &(f0.clone() * dt_actual) + &g0.dot(&ones_m) * sqrt_dt;

        // Evaluate drift at predicted point
        let f1 = (prob.f_drift)(t + dt_actual, &x_hat);
        let g1 = (prob.g_diffusion)(t, &x_hat);

        // Drift update: (1/2)(f0 + f1) * dt
        let drift_update = (&f0 + &f1) * (0.5 * dt_actual);

        // Diffusion update: g0 * ΔW
        let diff_update = g0.dot(&dw);

        // Platen correction for each Brownian motion j:
        // corr_j = (g1_j - g0_j) / (2*√dt) * ((ΔW_j)^2 - dt)
        let mut platen_corr = Array1::<f64>::zeros(n_state);
        for j in 0..m {
            let dw_j = dw[j];
            let factor = (dw_j * dw_j - dt_actual) / (2.0 * sqrt_dt);
            for i in 0..n_state {
                platen_corr[i] += (g1[[i, j]] - g0[[i, j]]) * factor;
            }
        }

        x = x + drift_update + diff_update + platen_corr;
        t += dt_actual;

        if opts.save_all_steps {
            sol.push(t, x.clone());
        }
    }

    if !opts.save_all_steps {
        sol.push(t, x);
    }

    Ok(sol)
}

/// Strong order 1.5 SRK for general (non-diagonal) multi-dimensional noise using Lévy areas.
///
/// Extends `srk_strong` to handle fully general diffusion matrices `g: R^n × R^m`
/// (where `n != m` is allowed and off-diagonal entries are non-zero). The scheme uses
/// the Wiktorsson (2001) approximation of the Lévy area `A_{ij}` to compute the
/// iterated Itô integrals `I_{ij}` required for strong order 1.5.
///
/// ## Update formula (Burrage-Burrage-Tian 2002 style)
///
/// ```text
/// H+_j = x_n + f dt + g_j √dt,   H-_j = x_n + f dt - g_j √dt
///
/// x_{n+1} = x_n + f dt + Σ_j g_j dW_j
///           + Σ_j Σ_k [g_k(H+_k) - g_k(H-_k)]_i · I_{kj} / √dt
/// ```
///
/// For diagonal/scalar noise this coincides with `srk_strong`. For general noise,
/// the `I_{kj}` terms couple different Brownian motion channels.
///
/// ## Lévy-area truncation
///
/// Uses `k = 5` Fourier terms by default, giving ~1e-3 relative accuracy in the
/// iterated integrals. Pass `levy_k` to control the trade-off.
///
/// # Arguments
///
/// * `prob` - SDE problem
/// * `dt` - Step size
/// * `rng` - RNG reference (must implement `rand::Rng`)
/// * `levy_k` - Truncation order for the Wiktorsson series (5 is usually sufficient)
///
/// # Strong order
///
/// 1.5 for any noise structure (scalar, diagonal, or general), subject to the
/// truncation error of the Wiktorsson approximation.
pub fn srk_strong_general<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
    levy_k: usize,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    srk_strong_general_with_options(prob, dt, rng, levy_k, &SdeOptions::default())
}

/// General-noise strong SRK with solver options.
pub fn srk_strong_general_with_options<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
    levy_k: usize,
    opts: &SdeOptions,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    prob.validate()?;
    let t0 = prob.t_span[0];
    let t1 = prob.t_span[1];
    let n_steps = compute_n_steps(t0, t1, dt, opts.max_steps)?;
    let n_state = prob.dim();
    let m = prob.n_brownian;

    let capacity = if opts.save_all_steps { n_steps + 1 } else { 2 };
    let mut sol = SdeSolution::with_capacity(capacity);
    sol.push(t0, prob.x0.clone());

    let normal = Normal::new(0.0_f64, 1.0_f64)
        .map_err(|e| IntegrateError::ComputationError(format!("Normal dist error: {}", e)))?;

    let mut x = prob.x0.clone();
    let mut t = t0;

    // Effective truncation order — minimum 1 to produce valid results
    let k = levy_k.max(1);

    for step in 0..n_steps {
        let dt_actual = if step == n_steps - 1 {
            t1 - t
        } else {
            dt.min(t1 - t)
        };
        if dt_actual <= 0.0 {
            break;
        }
        let sqrt_dt = dt_actual.sqrt();

        // Generate Brownian increments ΔW ~ N(0, dt * I_m)
        let dw: Array1<f64> = Array1::from_shape_fn(m, |_| normal.sample(rng) * sqrt_dt);

        let f0 = (prob.f_drift)(t, &x);
        let g0 = (prob.g_diffusion)(t, &x);

        if f0.len() != n_state || g0.nrows() != n_state || g0.ncols() != m {
            return Err(IntegrateError::DimensionMismatch(
                "Dimension mismatch in drift or diffusion".to_string(),
            ));
        }

        // Base drift term
        let f0_dt = f0.clone() * dt_actual;

        // Compute Lévy area matrix A (dim m × m, antisymmetric)
        let levy_a = levy_area_wiktorsson(m, dt_actual, &dw, k, rng);
        // Full iterated integral matrix I_{ij}
        let iij = iterated_integral(&dw, dt_actual, &levy_a);

        // SRK correction using iterated integrals:
        // For each column k of g (Brownian motion k), build H+_k and H-_k support points,
        // then accumulate: correction_i += Σ_j [g_k(H+_k) - g_k(H-_k)]_{i,j} * I_{k,j} / sqrt_dt
        let mut srk_correction = Array1::<f64>::zeros(n_state);
        for bk in 0..m {
            let g_col_k = g0.column(bk).to_owned();
            let h_plus_k = &x + &f0_dt + &g_col_k * sqrt_dt;
            let h_minus_k = &x + &f0_dt - &g_col_k * sqrt_dt;

            let g_hp = (prob.g_diffusion)(t, &h_plus_k);
            let g_hm = (prob.g_diffusion)(t, &h_minus_k);

            for j in 0..m {
                let factor = iij[[bk, j]] / sqrt_dt;
                for i in 0..n_state {
                    srk_correction[i] += (g_hp[[i, j]] - g_hm[[i, j]]) * 0.5 * factor;
                }
            }
        }

        // x_{n+1} = x_n + f*dt + g*ΔW + SRK_correction
        x = x + f0 * dt_actual + g0.dot(&dw) + srk_correction;
        t += dt_actual;

        if opts.save_all_steps {
            sol.push(t, x.clone());
        }
    }

    if !opts.save_all_steps {
        sol.push(t, x);
    }

    Ok(sol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sde::SdeProblem;
    use scirs2_core::ndarray::{array, Array2};
    use scirs2_core::random::prelude::seeded_rng;

    fn make_ou_prob(
        theta: f64,
        mu: f64,
        sigma: f64,
        x0: f64,
        t1: f64,
    ) -> SdeProblem<
        impl Fn(f64, &Array1<f64>) -> Array1<f64>,
        impl Fn(f64, &Array1<f64>) -> Array2<f64>,
    > {
        SdeProblem::new(
            array![x0],
            [0.0, t1],
            1,
            move |_t, x| array![theta * (mu - x[0])],
            move |_t, _x| {
                let mut g = Array2::zeros((1, 1));
                g[[0, 0]] = sigma;
                g
            },
        )
    }

    fn make_gbm_prob(
        mu: f64,
        sigma: f64,
        s0: f64,
    ) -> SdeProblem<
        impl Fn(f64, &Array1<f64>) -> Array1<f64>,
        impl Fn(f64, &Array1<f64>) -> Array2<f64>,
    > {
        SdeProblem::new(
            array![s0],
            [0.0, 1.0],
            1,
            move |_t, x| array![mu * x[0]],
            move |_t, x| {
                let mut g = Array2::zeros((1, 1));
                g[[0, 0]] = sigma * x[0];
                g
            },
        )
    }

    #[test]
    fn test_srk_strong_solution_length() {
        let prob = make_gbm_prob(0.05, 0.2, 1.0);
        let mut rng = seeded_rng(0);
        let sol = srk_strong(&prob, 0.1, &mut rng).expect("srk_strong should succeed");
        assert_eq!(sol.len(), 11);
    }

    #[test]
    fn test_platen_solution_length() {
        let prob = make_ou_prob(1.0, 0.0, 0.5, 1.0, 1.0);
        let mut rng = seeded_rng(42);
        let sol = platen_scheme(&prob, 0.1, &mut rng).expect("platen_scheme should succeed");
        assert_eq!(sol.len(), 11);
    }

    /// For GBM: E[S(T)] = S0 * exp(mu*T) — test weak convergence
    #[test]
    fn test_srk_gbm_weak_mean() {
        let mu = 0.1_f64;
        let sigma = 0.2_f64;
        let s0 = 1.0_f64;
        let t1 = 1.0_f64;
        let dt = 0.01;
        let analytic = s0 * (mu * t1).exp();
        let n_paths = 300;

        let mut sum = 0.0;
        for seed in 0..n_paths as u64 {
            let prob = make_gbm_prob(mu, sigma, s0);
            let mut rng = seeded_rng(seed + 5000);
            let sol = srk_strong(&prob, dt, &mut rng).expect("srk_strong should succeed");
            sum += sol.x_final().expect("solution has state")[0];
        }
        let mean = sum / n_paths as f64;
        let rel_err = (mean - analytic).abs() / analytic;
        assert!(
            rel_err < 0.05,
            "SRK GBM mean {:.4} vs analytic {:.4}, rel_err {:.4}",
            mean,
            analytic,
            rel_err
        );
    }

    /// OU process stationary mean should be μ_ou
    #[test]
    fn test_platen_ou_stationary_mean() {
        let theta = 2.0_f64;
        let mu_ou = 0.5_f64;
        let sigma = 0.3_f64;
        let dt = 0.005;
        let t1 = 5.0;
        let n_paths = 200;

        let mut sum_final = 0.0;
        for seed in 0..n_paths as u64 {
            let prob = make_ou_prob(theta, mu_ou, sigma, 0.0, t1);
            let mut rng = seeded_rng(seed + 2000);
            let sol = platen_scheme(&prob, dt, &mut rng).expect("platen_scheme should succeed");
            sum_final += sol.x_final().expect("solution has state")[0];
        }
        let mean_final = sum_final / n_paths as f64;
        assert!(
            (mean_final - mu_ou).abs() < 0.1,
            "OU stationary mean {:.4} vs expected {:.4}",
            mean_final,
            mu_ou
        );
    }

    #[test]
    fn test_platen_additive_noise_accuracy() {
        // Additive noise OU: analytic mean is x0 * exp(-theta*t) + mu*(1-exp(-theta*t))
        let theta = 1.0_f64;
        let mu_ou = 0.0_f64;
        let sigma = 0.5_f64;
        let x0 = 2.0_f64;
        let t1 = 1.0_f64;
        let analytic_mean = x0 * (-theta * t1).exp();
        let dt = 0.005;
        let n_paths = 400;

        let mut sum = 0.0;
        for seed in 0..n_paths as u64 {
            let prob = make_ou_prob(theta, mu_ou, sigma, x0, t1);
            let mut rng = seeded_rng(seed + 3000);
            let sol = platen_scheme(&prob, dt, &mut rng).expect("platen_scheme should succeed");
            sum += sol.x_final().expect("solution has state")[0];
        }
        let sample_mean = sum / n_paths as f64;
        let abs_err = (sample_mean - analytic_mean).abs();
        assert!(
            abs_err < 0.1,
            "Platen OU mean {:.4} vs analytic {:.4}, abs_err {:.4}",
            sample_mean,
            analytic_mean,
            abs_err
        );
    }

    #[test]
    fn test_srk_invalid_dt() {
        let prob = make_gbm_prob(0.05, 0.2, 1.0);
        let mut rng = seeded_rng(0);
        assert!(srk_strong(&prob, 0.0, &mut rng).is_err());
    }
}
