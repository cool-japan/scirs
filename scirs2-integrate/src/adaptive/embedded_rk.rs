//! Embedded Runge-Kutta Methods with adaptive step size control.
//!
//! This module provides high-quality embedded RK pairs:
//! - Dormand-Prince (DOPRI5): 4th/5th order, 7 evaluations per step
//! - Bogacki-Shampine (BS23): 2nd/3rd order, 3(4) evaluations per step (FSAL)
//! - Cash-Karp (RKCK): 4th/5th order, 6 evaluations per step

use crate::error::{IntegrateError, IntegrateResult};

/// Result of an ODE integration using embedded RK methods.
#[derive(Debug, Clone)]
pub struct OdeResult {
    /// Time points where the solution was recorded.
    pub t: Vec<f64>,
    /// Solution vectors at each time point; `y[i]` corresponds to `t[i]`.
    pub y: Vec<Vec<f64>>,
    /// Total number of accepted steps.
    pub n_steps: usize,
    /// Total number of rejected steps.
    pub n_rejected: usize,
    /// Total number of right-hand-side function evaluations.
    pub n_evals: usize,
}

/// Dense output interpolant for DOPRI5 (4th-order continuous extension).
///
/// Given two accepted steps at `t_prev`/`y_prev` and `t_curr`/`y_curr`, together
/// with the stage derivatives `k1`–`k7`, this evaluates the 4th-order Hermite
/// interpolant at any `theta = (t - t_prev) / h` in `[0, 1]`.
#[derive(Debug, Clone)]
pub struct Dopri5DenseOutput {
    /// Start of interval.
    pub t_prev: f64,
    /// Step size.
    pub h: f64,
    /// Interpolation coefficients (one per state dimension).
    pub coeffs: Vec<[f64; 5]>,
}

impl Dopri5DenseOutput {
    /// Evaluate the interpolant at absolute time `t`.
    pub fn eval(&self, t: f64) -> Vec<f64> {
        let theta = (t - self.t_prev) / self.h;
        let t1 = 1.0 - theta;
        self.coeffs
            .iter()
            .map(|c| c[0] + theta * (c[1] + t1 * (c[2] + theta * (c[3] + t1 * (c[4] * theta)))))
            .collect()
    }
}

// ─── DOPRI5 ──────────────────────────────────────────────────────────────────

/// Butcher tableau for Dormand-Prince (DOPRI5).
///
/// Standard coefficients from Dormand & Prince (1980).
mod dopri5_tableau {
    // Row 2
    pub const A21: f64 = 1.0 / 5.0;
    // Row 3
    pub const A31: f64 = 3.0 / 40.0;
    pub const A32: f64 = 9.0 / 40.0;
    // Row 4
    pub const A41: f64 = 44.0 / 45.0;
    pub const A42: f64 = -56.0 / 15.0;
    pub const A43: f64 = 32.0 / 9.0;
    // Row 5
    pub const A51: f64 = 19372.0 / 6561.0;
    pub const A52: f64 = -25360.0 / 2187.0;
    pub const A53: f64 = 64448.0 / 6561.0;
    pub const A54: f64 = -212.0 / 729.0;
    // Row 6
    pub const A61: f64 = 9017.0 / 3168.0;
    pub const A62: f64 = -355.0 / 33.0;
    pub const A63: f64 = 46732.0 / 5247.0;
    pub const A64: f64 = 49.0 / 176.0;
    pub const A65: f64 = -5103.0 / 18656.0;
    // Row 7 (=5th-order weights b)
    pub const A71: f64 = 35.0 / 384.0;
    pub const A72: f64 = 0.0;
    pub const A73: f64 = 500.0 / 1113.0;
    pub const A74: f64 = 125.0 / 192.0;
    pub const A75: f64 = -2187.0 / 6784.0;
    pub const A76: f64 = 11.0 / 84.0;

    // c nodes
    pub const C2: f64 = 1.0 / 5.0;
    pub const C3: f64 = 3.0 / 10.0;
    pub const C4: f64 = 4.0 / 5.0;
    pub const C5: f64 = 8.0 / 9.0;
    // C6 = 1, C7 = 1

    // 5th-order weights (same as A7x above)
    pub const B5_1: f64 = A71;
    pub const B5_3: f64 = A73;
    pub const B5_4: f64 = A74;
    pub const B5_5: f64 = A75;
    pub const B5_6: f64 = A76;

    // 4th-order weights (local extrapolation variant)
    pub const B4_1: f64 = 5179.0 / 57600.0;
    pub const B4_3: f64 = 7571.0 / 16695.0;
    pub const B4_4: f64 = 393.0 / 640.0;
    pub const B4_5: f64 = -92097.0 / 339200.0;
    pub const B4_6: f64 = 187.0 / 2100.0;
    pub const B4_7: f64 = 1.0 / 40.0;

    // Dense output coefficients (continuous extension from Shampine 1986)
    pub const D1: f64 = -12715105075.0 / 11282082432.0;
    pub const D3: f64 = 87487479700.0 / 32700410799.0;
    pub const D4: f64 = -10690763975.0 / 1880347072.0;
    pub const D5: f64 = 701980252875.0 / 199316789632.0;
    pub const D6: f64 = -1453857185.0 / 822651844.0;
    pub const D7: f64 = 69997945.0 / 29380423.0;
}

/// Perform a single DOPRI5 step.
///
/// Returns `(y_5th, y_4th, k1_new, evals)`.
fn dopri5_step<F>(
    f: &F,
    t: f64,
    y: &[f64],
    h: f64,
    k1: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize)
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    use dopri5_tableau::*;
    let n = y.len();

    // k2
    let y2: Vec<f64> = (0..n).map(|i| y[i] + h * A21 * k1[i]).collect();
    let k2 = f(t + C2 * h, &y2);

    // k3
    let y3: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (A31 * k1[i] + A32 * k2[i]))
        .collect();
    let k3 = f(t + C3 * h, &y3);

    // k4
    let y4: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]))
        .collect();
    let k4 = f(t + C4 * h, &y4);

    // k5
    let y5: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]))
        .collect();
    let k5 = f(t + C5 * h, &y5);

    // k6
    let y6: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]))
        .collect();
    let k6 = f(t + h, &y6);

    // 5th-order solution (used as propagating solution with local extrapolation)
    let y_5th: Vec<f64> = (0..n)
        .map(|i| {
            y[i] + h * (B5_1 * k1[i] + B5_3 * k3[i] + B5_4 * k4[i] + B5_5 * k5[i] + B5_6 * k6[i])
        })
        .collect();

    // k7 (= f at new point, needed for FSAL-like dense output)
    let k7 = f(t + h, &y_5th);

    // 4th-order solution for error estimate
    let y_4th: Vec<f64> = (0..n)
        .map(|i| {
            y[i] + h
                * (B4_1 * k1[i]
                    + B4_3 * k3[i]
                    + B4_4 * k4[i]
                    + B4_5 * k5[i]
                    + B4_6 * k6[i]
                    + B4_7 * k7[i])
        })
        .collect();

    // 6 new evaluations (k2..k7), k1 is reused from prior step
    (y_5th, y_4th, k7, 6)
}

/// Compute the normalised error for a DOPRI5 step.
fn mixed_error_norm(y_old: &[f64], y_new: &[f64], y_err: &[f64], rtol: f64, atol: f64) -> f64 {
    let n = y_old.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = y_old
        .iter()
        .zip(y_new.iter())
        .zip(y_err.iter())
        .map(|((yo, yn), ye)| {
            let sc = atol + rtol * yo.abs().max(yn.abs());
            (ye / sc).powi(2)
        })
        .sum();
    (sum / n as f64).sqrt()
}

/// Build the dense output interpolant for one DOPRI5 step.
fn build_dense_output(
    t_prev: f64,
    h: f64,
    y_prev: &[f64],
    y_curr: &[f64],
    k1: &[f64],
    k3: &[f64],
    k4: &[f64],
    k5: &[f64],
    k6: &[f64],
    k7: &[f64],
) -> Dopri5DenseOutput {
    use dopri5_tableau::*;
    let n = y_prev.len();
    // Shampine's 4th-order continuous extension
    let coeffs: Vec<[f64; 5]> = (0..n)
        .map(|i| {
            let dy = y_curr[i] - y_prev[i];
            // Coefficients for the polynomial in theta
            let c0 = y_prev[i];
            let c1 = dy;
            let c2 = h * k1[i] - dy;
            let c3 = 2.0 * dy - h * (k1[i] + k7[i]);
            let c4 =
                h * (D1 * k1[i] + D3 * k3[i] + D4 * k4[i] + D5 * k5[i] + D6 * k6[i] + D7 * k7[i]);
            [c0, c1, c2, c3, c4]
        })
        .collect();
    Dopri5DenseOutput { t_prev, h, coeffs }
}

/// Solve an ODE using the Dormand-Prince (DOPRI5) method.
///
/// DOPRI5 uses a 7-stage, 5th-order method with an embedded 4th-order
/// solution for step size control. The integrator uses local extrapolation:
/// the 5th-order solution is propagated and the 4th-order solution is used
/// only for error estimation.
///
/// # Arguments
///
/// * `f`     – The right-hand side `dy/dt = f(t, y)`.
/// * `t0`    – Initial time.
/// * `y0`    – Initial state vector.
/// * `t_end` – Final time.
/// * `rtol`  – Relative tolerance (per-component mixed norm).
/// * `atol`  – Absolute tolerance (per-component mixed norm).
///
/// # Errors
///
/// Returns [`IntegrateError::StepSizeTooSmall`] if the step size becomes
/// too small to make progress, or [`IntegrateError::ComputationError`] if
/// the maximum number of steps is exceeded.
pub fn dopri5<F>(
    f: F,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    rtol: f64,
    atol: f64,
) -> IntegrateResult<OdeResult>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    const SAFETY: f64 = 0.9;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    const MAX_STEPS: usize = 100_000;
    const ORDER: f64 = 5.0; // 5th-order method

    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();

    // Initial step size heuristic
    let span = (t_end - t0).abs();
    let mut h = span * 1e-3;
    if h == 0.0 {
        h = 1e-6;
    }
    let h_min = span * 1e-12;
    let h_max = span * 0.1;
    let h_sign = if t_end >= t0 { 1.0_f64 } else { -1.0 };
    h = h_sign * h.abs();

    let mut t_out = vec![t0];
    let mut y_out = vec![y0.to_vec()];
    let mut n_steps: usize = 0;
    let mut n_rejected: usize = 0;
    let mut n_evals: usize = 1; // first k1

    // Initial k1
    let mut k1 = f(t, &y);

    loop {
        if (h_sign > 0.0 && t >= t_end) || (h_sign < 0.0 && t <= t_end) {
            break;
        }
        if n_steps >= MAX_STEPS {
            return Err(IntegrateError::ComputationError(format!(
                "DOPRI5: maximum step count ({MAX_STEPS}) exceeded at t={t}"
            )));
        }

        // Clip step to not overshoot
        if (h_sign > 0.0 && t + h > t_end) || (h_sign < 0.0 && t + h < t_end) {
            h = t_end - t;
        }

        if h.abs() < h_min {
            return Err(IntegrateError::StepSizeTooSmall(format!(
                "DOPRI5: step size {h} < minimum {h_min} at t={t}"
            )));
        }

        let (y5, y4, k7, evals) = dopri5_step(&f, t, &y, h, &k1);
        n_evals += evals;

        // Error vector = y5 - y4
        let y_err: Vec<f64> = (0..n).map(|i| y5[i] - y4[i]).collect();
        let err = mixed_error_norm(&y, &y5, &y_err, rtol, atol);

        if err <= 1.0 {
            // Step accepted
            n_steps += 1;
            t += h;
            y = y5.clone();
            k1 = k7;
            t_out.push(t);
            y_out.push(y.clone());

            // Increase step size
            let factor = if err == 0.0 {
                MAX_FACTOR
            } else {
                (SAFETY * err.powf(-1.0 / ORDER)).clamp(MIN_FACTOR, MAX_FACTOR)
            };
            h = h_sign * (h.abs() * factor).min(h_max);
        } else {
            // Step rejected
            n_rejected += 1;
            let factor = (SAFETY * err.powf(-1.0 / ORDER)).max(MIN_FACTOR);
            h = h_sign * (h.abs() * factor).min(h_max);
        }
    }

    Ok(OdeResult {
        t: t_out,
        y: y_out,
        n_steps,
        n_rejected,
        n_evals,
    })
}

/// Solve an ODE using the Dormand-Prince (DOPRI5) method and also return
/// dense output interpolants for each accepted step.
pub fn dopri5_dense<F>(
    f: F,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    rtol: f64,
    atol: f64,
) -> IntegrateResult<(OdeResult, Vec<Dopri5DenseOutput>)>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    const SAFETY: f64 = 0.9;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    const MAX_STEPS: usize = 100_000;
    const ORDER: f64 = 5.0;

    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();

    let span = (t_end - t0).abs();
    let mut h = span * 1e-3;
    if h == 0.0 {
        h = 1e-6;
    }
    let h_min = span * 1e-12;
    let h_max = span * 0.1;
    let h_sign = if t_end >= t0 { 1.0_f64 } else { -1.0 };
    h = h_sign * h.abs();

    let mut t_out = vec![t0];
    let mut y_out = vec![y0.to_vec()];
    let mut dense_out: Vec<Dopri5DenseOutput> = Vec::new();
    let mut n_steps: usize = 0;
    let mut n_rejected: usize = 0;
    let mut n_evals: usize = 1;

    let mut k1 = f(t, &y);

    loop {
        if (h_sign > 0.0 && t >= t_end) || (h_sign < 0.0 && t <= t_end) {
            break;
        }
        if n_steps >= MAX_STEPS {
            return Err(IntegrateError::ComputationError(format!(
                "DOPRI5: maximum step count ({MAX_STEPS}) exceeded at t={t}"
            )));
        }

        if (h_sign > 0.0 && t + h > t_end) || (h_sign < 0.0 && t + h < t_end) {
            h = t_end - t;
        }

        if h.abs() < h_min {
            return Err(IntegrateError::StepSizeTooSmall(format!(
                "DOPRI5: step size {h} < minimum {h_min} at t={t}"
            )));
        }

        // We need intermediate stages for dense output; inline the step
        let (y5, y4, k7, evals, stage_k3, stage_k4, stage_k5, stage_k6) =
            dopri5_step_with_stages(&f, t, &y, h, &k1);
        n_evals += evals;

        let y_err: Vec<f64> = (0..n).map(|i| y5[i] - y4[i]).collect();
        let err = mixed_error_norm(&y, &y5, &y_err, rtol, atol);

        if err <= 1.0 {
            // Build dense output before updating y
            let dense = build_dense_output(
                t, h, &y, &y5, &k1, &stage_k3, &stage_k4, &stage_k5, &stage_k6, &k7,
            );
            dense_out.push(dense);

            n_steps += 1;
            t += h;
            y = y5.clone();
            k1 = k7;
            t_out.push(t);
            y_out.push(y.clone());

            let factor = if err == 0.0 {
                MAX_FACTOR
            } else {
                (SAFETY * err.powf(-1.0 / ORDER)).clamp(MIN_FACTOR, MAX_FACTOR)
            };
            h = h_sign * (h.abs() * factor).min(h_max);
        } else {
            n_rejected += 1;
            let factor = (SAFETY * err.powf(-1.0 / ORDER)).max(MIN_FACTOR);
            h = h_sign * (h.abs() * factor).min(h_max);
        }
    }

    Ok((
        OdeResult {
            t: t_out,
            y: y_out,
            n_steps,
            n_rejected,
            n_evals,
        },
        dense_out,
    ))
}

/// Like `dopri5_step` but also returns intermediate stages for dense output.
#[allow(clippy::type_complexity)]
fn dopri5_step_with_stages<F>(
    f: &F,
    t: f64,
    y: &[f64],
    h: f64,
    k1: &[f64],
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    usize,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    use dopri5_tableau::*;
    let n = y.len();

    let y2: Vec<f64> = (0..n).map(|i| y[i] + h * A21 * k1[i]).collect();
    let k2 = f(t + C2 * h, &y2);

    let y3: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (A31 * k1[i] + A32 * k2[i]))
        .collect();
    let k3 = f(t + C3 * h, &y3);

    let y4: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]))
        .collect();
    let k4 = f(t + C4 * h, &y4);

    let y5: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]))
        .collect();
    let k5 = f(t + C5 * h, &y5);

    let y6: Vec<f64> = (0..n)
        .map(|i| y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]))
        .collect();
    let k6 = f(t + h, &y6);

    let y_5th: Vec<f64> = (0..n)
        .map(|i| {
            y[i] + h * (B5_1 * k1[i] + B5_3 * k3[i] + B5_4 * k4[i] + B5_5 * k5[i] + B5_6 * k6[i])
        })
        .collect();

    let k7 = f(t + h, &y_5th);

    let y_4th: Vec<f64> = (0..n)
        .map(|i| {
            y[i] + h
                * (B4_1 * k1[i]
                    + B4_3 * k3[i]
                    + B4_4 * k4[i]
                    + B4_5 * k5[i]
                    + B4_6 * k6[i]
                    + B4_7 * k7[i])
        })
        .collect();

    (y_5th, y_4th, k7, 6, k3, k4, k5, k6)
}

// ─── BOGACKI-SHAMPINE (BS23) ──────────────────────────────────────────────────

/// Butcher tableau for Bogacki-Shampine (BS23).
///
/// This is a FSAL (First Same As Last) method, so only 3 new evaluations
/// are needed per accepted step.
mod bs23_tableau {
    // Nodes
    pub const C2: f64 = 1.0 / 2.0;
    pub const C3: f64 = 3.0 / 4.0;

    // Stage coefficients
    pub const A21: f64 = 1.0 / 2.0;
    pub const A31: f64 = 0.0;
    pub const A32: f64 = 3.0 / 4.0;
    pub const A41: f64 = 2.0 / 9.0;
    pub const A42: f64 = 1.0 / 3.0;
    pub const A43: f64 = 4.0 / 9.0;

    // 3rd-order weights (same as A4x)
    pub const B3_1: f64 = A41;
    pub const B3_2: f64 = A42;
    pub const B3_3: f64 = A43;

    // 2nd-order weights
    pub const B2_1: f64 = 7.0 / 24.0;
    pub const B2_2: f64 = 1.0 / 4.0;
    pub const B2_3: f64 = 1.0 / 3.0;
    pub const B2_4: f64 = 1.0 / 8.0;
}

/// Solve an ODE using the Bogacki-Shampine (BS23) method.
///
/// BS23 is a 2nd/3rd-order FSAL embedded pair well-suited for mildly
/// non-stiff problems where low accuracy is acceptable (similar to
/// SciPy's `RK23`).
///
/// # Arguments
///
/// * `f`     – The right-hand side `dy/dt = f(t, y)`.
/// * `t0`    – Initial time.
/// * `y0`    – Initial state vector.
/// * `t_end` – Final time.
/// * `rtol`  – Relative tolerance.
/// * `atol`  – Absolute tolerance.
pub fn bs23<F>(
    f: F,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    rtol: f64,
    atol: f64,
) -> IntegrateResult<OdeResult>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    use bs23_tableau::*;

    const SAFETY: f64 = 0.9;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    const MAX_STEPS: usize = 100_000;
    const ORDER: f64 = 3.0;

    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();

    let span = (t_end - t0).abs();
    let mut h = span * 1e-2;
    if h == 0.0 {
        h = 1e-6;
    }
    let h_min = span * 1e-12;
    let h_max = span * 0.5;
    let h_sign = if t_end >= t0 { 1.0_f64 } else { -1.0 };
    h = h_sign * h.abs();

    let mut t_out = vec![t0];
    let mut y_out = vec![y0.to_vec()];
    let mut n_steps: usize = 0;
    let mut n_rejected: usize = 0;
    let mut n_evals: usize = 1;

    // FSAL: k1 is f(t, y)
    let mut k1 = f(t, &y);

    loop {
        if (h_sign > 0.0 && t >= t_end) || (h_sign < 0.0 && t <= t_end) {
            break;
        }
        if n_steps >= MAX_STEPS {
            return Err(IntegrateError::ComputationError(format!(
                "BS23: maximum step count ({MAX_STEPS}) exceeded at t={t}"
            )));
        }

        if (h_sign > 0.0 && t + h > t_end) || (h_sign < 0.0 && t + h < t_end) {
            h = t_end - t;
        }

        if h.abs() < h_min {
            return Err(IntegrateError::StepSizeTooSmall(format!(
                "BS23: step size {h} < minimum {h_min} at t={t}"
            )));
        }

        // Stage 2
        let y2: Vec<f64> = (0..n).map(|i| y[i] + h * A21 * k1[i]).collect();
        let k2 = f(t + C2 * h, &y2);

        // Stage 3
        let y3: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (A31 * k1[i] + A32 * k2[i]))
            .collect();
        let k3 = f(t + C3 * h, &y3);

        // 3rd-order solution
        let y3rd: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (B3_1 * k1[i] + B3_2 * k2[i] + B3_3 * k3[i]))
            .collect();

        // Stage 4 at 3rd-order endpoint (FSAL)
        let k4 = f(t + h, &y3rd);
        n_evals += 3; // k2, k3, k4

        // 2nd-order solution
        let y2nd: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (B2_1 * k1[i] + B2_2 * k2[i] + B2_3 * k3[i] + B2_4 * k4[i]))
            .collect();

        let y_err: Vec<f64> = (0..n).map(|i| y3rd[i] - y2nd[i]).collect();
        let err = mixed_error_norm(&y, &y3rd, &y_err, rtol, atol);

        if err <= 1.0 {
            n_steps += 1;
            t += h;
            y = y3rd;
            k1 = k4; // FSAL reuse
            t_out.push(t);
            y_out.push(y.clone());

            let factor = if err == 0.0 {
                MAX_FACTOR
            } else {
                (SAFETY * err.powf(-1.0 / ORDER)).clamp(MIN_FACTOR, MAX_FACTOR)
            };
            h = h_sign * (h.abs() * factor).min(h_max);
        } else {
            n_rejected += 1;
            let factor = (SAFETY * err.powf(-1.0 / ORDER)).max(MIN_FACTOR);
            h = h_sign * (h.abs() * factor).min(h_max);
            // Must recompute k1 since step was rejected and we may have moved
            // (we haven't moved t, so k1 is still valid; FSAL works only on acceptance)
        }
    }

    Ok(OdeResult {
        t: t_out,
        y: y_out,
        n_steps,
        n_rejected,
        n_evals,
    })
}

// ─── CASH-KARP (RKCK) ────────────────────────────────────────────────────────

/// Butcher tableau for Cash-Karp (RKCK) 4th/5th-order pair.
mod rkck_tableau {
    // Nodes
    pub const C2: f64 = 1.0 / 5.0;
    pub const C3: f64 = 3.0 / 10.0;
    pub const C4: f64 = 3.0 / 5.0;
    // C5 = 1.0
    pub const C6: f64 = 7.0 / 8.0;

    // Stage matrix
    pub const A21: f64 = 1.0 / 5.0;

    pub const A31: f64 = 3.0 / 40.0;
    pub const A32: f64 = 9.0 / 40.0;

    pub const A41: f64 = 3.0 / 10.0;
    pub const A42: f64 = -9.0 / 10.0;
    pub const A43: f64 = 6.0 / 5.0;

    pub const A51: f64 = -11.0 / 54.0;
    pub const A52: f64 = 5.0 / 2.0;
    pub const A53: f64 = -70.0 / 27.0;
    pub const A54: f64 = 35.0 / 27.0;

    pub const A61: f64 = 1631.0 / 55296.0;
    pub const A62: f64 = 175.0 / 512.0;
    pub const A63: f64 = 575.0 / 13824.0;
    pub const A64: f64 = 44275.0 / 110592.0;
    pub const A65: f64 = 253.0 / 4096.0;

    // 5th-order weights
    pub const B5_1: f64 = 37.0 / 378.0;
    pub const B5_3: f64 = 250.0 / 621.0;
    pub const B5_4: f64 = 125.0 / 594.0;
    pub const B5_6: f64 = 512.0 / 1771.0;

    // 4th-order weights
    pub const B4_1: f64 = 2825.0 / 27648.0;
    pub const B4_3: f64 = 18575.0 / 48384.0;
    pub const B4_4: f64 = 13525.0 / 55296.0;
    pub const B4_5: f64 = 277.0 / 14336.0;
    pub const B4_6: f64 = 1.0 / 4.0;
}

/// Solve an ODE using the Cash-Karp (RKCK) method.
///
/// RKCK uses a 4th/5th-order embedded pair with 6 function evaluations
/// per step. It is often a good general-purpose choice.
///
/// # Arguments
///
/// * `f`     – The right-hand side `dy/dt = f(t, y)`.
/// * `t0`    – Initial time.
/// * `y0`    – Initial state vector.
/// * `t_end` – Final time.
/// * `rtol`  – Relative tolerance.
/// * `atol`  – Absolute tolerance.
pub fn rkck<F>(
    f: F,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    rtol: f64,
    atol: f64,
) -> IntegrateResult<OdeResult>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    use rkck_tableau::*;

    const SAFETY: f64 = 0.9;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    const MAX_STEPS: usize = 100_000;
    const ORDER: f64 = 5.0;

    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();

    let span = (t_end - t0).abs();
    let mut h = span * 1e-3;
    if h == 0.0 {
        h = 1e-6;
    }
    let h_min = span * 1e-12;
    let h_max = span * 0.1;
    let h_sign = if t_end >= t0 { 1.0_f64 } else { -1.0 };
    h = h_sign * h.abs();

    let mut t_out = vec![t0];
    let mut y_out = vec![y0.to_vec()];
    let mut n_steps: usize = 0;
    let mut n_rejected: usize = 0;
    let mut n_evals: usize = 0;

    loop {
        if (h_sign > 0.0 && t >= t_end) || (h_sign < 0.0 && t <= t_end) {
            break;
        }
        if n_steps >= MAX_STEPS {
            return Err(IntegrateError::ComputationError(format!(
                "RKCK: maximum step count ({MAX_STEPS}) exceeded at t={t}"
            )));
        }

        if (h_sign > 0.0 && t + h > t_end) || (h_sign < 0.0 && t + h < t_end) {
            h = t_end - t;
        }

        if h.abs() < h_min {
            return Err(IntegrateError::StepSizeTooSmall(format!(
                "RKCK: step size {h} < minimum {h_min} at t={t}"
            )));
        }

        // Six stages
        let k1 = f(t, &y);
        let y2: Vec<f64> = (0..n).map(|i| y[i] + h * A21 * k1[i]).collect();
        let k2 = f(t + C2 * h, &y2);
        let y3: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (A31 * k1[i] + A32 * k2[i]))
            .collect();
        let k3 = f(t + C3 * h, &y3);
        let y4: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]))
            .collect();
        let k4 = f(t + C4 * h, &y4);
        let y5: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]))
            .collect();
        let k5 = f(t + h, &y5);
        let y6: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i])
            })
            .collect();
        let k6 = f(t + C6 * h, &y6);
        n_evals += 6;

        // 5th-order solution
        let y5th: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (B5_1 * k1[i] + B5_3 * k3[i] + B5_4 * k4[i] + B5_6 * k6[i]))
            .collect();

        // 4th-order solution
        let y4th: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + h
                    * (B4_1 * k1[i] + B4_3 * k3[i] + B4_4 * k4[i] + B4_5 * k5[i] + B4_6 * k6[i])
            })
            .collect();

        let y_err: Vec<f64> = (0..n).map(|i| y5th[i] - y4th[i]).collect();
        let err = mixed_error_norm(&y, &y5th, &y_err, rtol, atol);

        if err <= 1.0 {
            n_steps += 1;
            t += h;
            y = y5th;
            t_out.push(t);
            y_out.push(y.clone());

            let factor = if err == 0.0 {
                MAX_FACTOR
            } else {
                (SAFETY * err.powf(-1.0 / ORDER)).clamp(MIN_FACTOR, MAX_FACTOR)
            };
            h = h_sign * (h.abs() * factor).min(h_max);
        } else {
            n_rejected += 1;
            let factor = (SAFETY * err.powf(-1.0 / ORDER)).max(MIN_FACTOR);
            h = h_sign * (h.abs() * factor).min(h_max);
        }
    }

    Ok(OdeResult {
        t: t_out,
        y: y_out,
        n_steps,
        n_rejected,
        n_evals,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple exponential: dy/dt = -y, y(0) = 1 → y(t) = exp(-t)
    fn exp_decay(t: f64, y: &[f64]) -> Vec<f64> {
        let _ = t;
        vec![-y[0]]
    }

    /// Harmonic oscillator: dy1/dt = y2, dy2/dt = -y1
    fn harmonic(t: f64, y: &[f64]) -> Vec<f64> {
        let _ = t;
        vec![y[1], -y[0]]
    }

    #[test]
    fn dopri5_exp_decay() {
        let res = dopri5(exp_decay, 0.0, &[1.0], 5.0, 1e-8, 1e-10).expect("dopri5 failed");
        let t_end = *res.t.last().expect("empty result");
        let y_end = res.y.last().expect("empty result")[0];
        assert!((t_end - 5.0).abs() < 1e-12, "t_end mismatch: {t_end}");
        assert!(
            (y_end - (-5.0_f64).exp()).abs() < 1e-7,
            "y_end mismatch: {y_end}"
        );
        assert!(res.n_rejected == 0 || res.n_rejected < res.n_steps);
    }

    #[test]
    fn dopri5_harmonic() {
        let res =
            dopri5(harmonic, 0.0, &[1.0, 0.0], 10.0, 1e-8, 1e-10).expect("dopri5 harmonic failed");
        let y_end = &res.y.last().expect("empty")[..];
        // y1(t) = cos(t), y2(t) = -sin(t)
        assert!((y_end[0] - (10.0_f64).cos()).abs() < 1e-6);
    }

    #[test]
    fn bs23_exp_decay() {
        let res = bs23(exp_decay, 0.0, &[1.0], 5.0, 1e-6, 1e-8).expect("bs23 failed");
        let y_end = res.y.last().expect("empty")[0];
        assert!((y_end - (-5.0_f64).exp()).abs() < 1e-5);
    }

    #[test]
    fn rkck_exp_decay() {
        let res = rkck(exp_decay, 0.0, &[1.0], 5.0, 1e-7, 1e-9).expect("rkck failed");
        let y_end = res.y.last().expect("empty")[0];
        assert!((y_end - (-5.0_f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn dopri5_dense_output() {
        let (res, dense) = dopri5_dense(
            harmonic,
            0.0,
            &[1.0, 0.0],
            2.0 * std::f64::consts::PI,
            1e-8,
            1e-10,
        )
        .expect("dopri5_dense failed");
        // Evaluate at t = pi/2
        let t_query = std::f64::consts::FRAC_PI_2;
        // Find the step containing t_query
        let idx = res
            .t
            .windows(2)
            .position(|w| w[0] <= t_query && t_query <= w[1]);
        if let Some(i) = idx {
            let y_interp = dense[i].eval(t_query);
            // y1(pi/2) = cos(pi/2) ≈ 0
            assert!(
                y_interp[0].abs() < 1e-5,
                "dense y1 at pi/2: {}",
                y_interp[0]
            );
        }
    }
}
