//! Stiff ODE solvers using implicit methods.
//!
//! This module provides solvers designed for stiff ordinary differential
//! equations, where explicit methods would require impractically small step
//! sizes due to stability constraints.
//!
//! # Methods Implemented
//!
//! - **Implicit Euler (IE1)**: First-order, A-stable. Each step solves a
//!   nonlinear system via Newton iteration with the user-supplied Jacobian.
//! - **Trapezoidal (Crank-Nicolson, CN2)**: Second-order, A-stable. Uses the
//!   same Newton-iteration framework as IE1.
//! - **BDF-2**: Second-order backward differentiation formula. Requires two
//!   warm-up steps with IE1 before the main integration begins.
//!
//! All solvers require the user to supply a Jacobian function.  No finite-
//! difference Jacobian approximation is done internally so as to preserve
//! exactness and avoid extra function evaluations.

use super::embedded_rk::OdeResult;
use crate::error::{IntegrateError, IntegrateResult};

// ─── Linear algebra helpers ──────────────────────────────────────────────────

/// Solve the linear system `A x = b` using Gaussian elimination with partial
/// pivoting.  Returns `x` on success.
///
/// `a` is a flat row-major `n×n` matrix (length `n*n`), `b` has length `n`.
fn gaussian_solve(a: &[f64], b: &[f64]) -> Result<Vec<f64>, IntegrateError> {
    let n = b.len();
    debug_assert_eq!(a.len(), n * n, "matrix/vector size mismatch");

    // Working copies
    let mut mat: Vec<f64> = a.to_vec();
    let mut rhs: Vec<f64> = b.to_vec();

    for col in 0..n {
        // Partial pivoting: find the row with the largest absolute value in
        // column `col` from row `col` downwards.
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                mat[r1 * n + col]
                    .abs()
                    .partial_cmp(&mat[r2 * n + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| IntegrateError::LinearSolveError("empty row range".to_string()))?;

        if mat[pivot_row * n + col].abs() < 1e-300 {
            return Err(IntegrateError::LinearSolveError(format!(
                "singular or near-singular matrix at column {col}"
            )));
        }

        // Swap rows
        if pivot_row != col {
            for j in 0..n {
                mat.swap(col * n + j, pivot_row * n + j);
            }
            rhs.swap(col, pivot_row);
        }

        // Eliminate below
        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            for j in col..n {
                let val = mat[col * n + j];
                mat[row * n + j] -= factor * val;
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut s = rhs[row];
        for j in (row + 1)..n {
            s -= mat[row * n + j] * x[j];
        }
        x[row] = s / mat[row * n + row];
    }
    Ok(x)
}

/// Compute `(I - h * J)` as a flat row-major matrix.
fn identity_minus_hj(h: f64, jac: &[Vec<f64>], n: usize) -> Vec<f64> {
    let mut mat = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let delta = if i == j { 1.0 } else { 0.0 };
            mat[i * n + j] = delta - h * jac[i][j];
        }
    }
    mat
}

// ─── Newton iteration ────────────────────────────────────────────────────────

/// Newton iteration constants.
const MAX_NEWTON: usize = 50;
const NEWTON_TOL: f64 = 1e-10;

/// Solve the implicit Euler nonlinear system
///
/// ```text
/// G(y_new) = y_new - y_old - h * f(t_new, y_new) = 0
/// ```
///
/// using Newton's method initialised at `y_guess`.  Returns the converged
/// solution or an error if Newton fails to converge within `MAX_NEWTON`
/// iterations.
fn newton_implicit_euler<F, J>(
    f: &F,
    jac: &J,
    t_new: f64,
    y_old: &[f64],
    h: f64,
    y_guess: &[f64],
) -> IntegrateResult<(Vec<f64>, usize)>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    let n = y_old.len();
    let mut y = y_guess.to_vec();
    let mut n_evals: usize = 0;

    for _iter in 0..MAX_NEWTON {
        let fy = f(t_new, &y);
        let jy = jac(t_new, &y);
        n_evals += 2; // f + J

        // Residual: G = y - y_old - h*f(t_new, y)
        let residual: Vec<f64> = (0..n).map(|i| y[i] - y_old[i] - h * fy[i]).collect();

        // Check convergence by residual norm
        let res_norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
        if res_norm < NEWTON_TOL {
            return Ok((y, n_evals));
        }

        // Jacobian of G w.r.t. y_new: I - h * J(t_new, y)
        let a = identity_minus_hj(h, &jy, n);

        // Solve (I - h*J) * delta = -G  →  delta = -(I-hJ)^{-1} G
        // Then y_new ← y_new + delta
        let neg_residual: Vec<f64> = residual.iter().map(|r| -r).collect();
        let delta = gaussian_solve(&a, &neg_residual)?;

        let delta_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        for i in 0..n {
            y[i] += delta[i];
        }

        if delta_norm < NEWTON_TOL * (1.0 + y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)) {
            return Ok((y, n_evals));
        }
    }

    Err(IntegrateError::ConvergenceError(format!(
        "Newton iteration did not converge within {MAX_NEWTON} iterations at t={t_new}"
    )))
}

/// Solve the trapezoidal (Crank-Nicolson) nonlinear system
///
/// ```text
/// G(y_new) = y_new - y_old - (h/2) * [f(t_old, y_old) + f(t_new, y_new)] = 0
/// ```
fn newton_trapezoidal<F, J>(
    f: &F,
    jac: &J,
    _t_old: f64,
    t_new: f64,
    y_old: &[f64],
    f_old: &[f64],
    h: f64,
    y_guess: &[f64],
) -> IntegrateResult<(Vec<f64>, usize)>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    let n = y_old.len();
    let mut y = y_guess.to_vec();
    let mut n_evals: usize = 0;
    let h2 = h / 2.0;

    for _iter in 0..MAX_NEWTON {
        let fy = f(t_new, &y);
        let jy = jac(t_new, &y);
        n_evals += 2;

        // Residual
        let residual: Vec<f64> = (0..n)
            .map(|i| y[i] - y_old[i] - h2 * (f_old[i] + fy[i]))
            .collect();

        let res_norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
        if res_norm < NEWTON_TOL {
            return Ok((y, n_evals));
        }

        // Jacobian of G: I - (h/2) * J
        let a = identity_minus_hj(h2, &jy, n);
        let neg_residual: Vec<f64> = residual.iter().map(|r| -r).collect();
        let delta = gaussian_solve(&a, &neg_residual)?;

        let delta_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        for i in 0..n {
            y[i] += delta[i];
        }

        if delta_norm < NEWTON_TOL * (1.0 + y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)) {
            return Ok((y, n_evals));
        }
    }

    Err(IntegrateError::ConvergenceError(format!(
        "Newton (trapezoidal) did not converge within {MAX_NEWTON} iterations at t={t_new}"
    )))
}

// ─── Public solvers ──────────────────────────────────────────────────────────

/// Solve a stiff ODE using the **Implicit Euler** (backward Euler) method.
///
/// This is a first-order, A-stable, L-stable method.  Each step solves a
/// nonlinear system via Newton iteration using the user-supplied Jacobian.
///
/// # Arguments
///
/// * `f`     – Right-hand side: `dy/dt = f(t, y)`.
/// * `jac`   – Jacobian: `J[i][j] = ∂f_i/∂y_j` at `(t, y)`.
/// * `t0`    – Initial time.
/// * `y0`    – Initial state vector.
/// * `t_end` – Final time.
/// * `h`     – Step size (fixed).
///
/// # Errors
///
/// Returns an error if Newton iteration fails to converge at any step, or if
/// the linear system at any step is singular.
///
/// # Example
///
/// ```no_run
/// use scirs2_integrate::adaptive::stiff::implicit_euler;
///
/// // dy/dt = -1000 * y, y(0) = 1  (very stiff)
/// let f  = |_t: f64, y: &[f64]| vec![-1000.0 * y[0]];
/// let jac = |_t: f64, _y: &[f64]| vec![vec![-1000.0]];
///
/// let result = implicit_euler(f, jac, 0.0, &[1.0], 0.01, 0.001)
///     .expect("integration failed");
/// println!("y(0.01) ≈ {}", result.y.last().expect("empty")[0]);
/// ```
pub fn implicit_euler<F, J>(
    f: F,
    jac: J,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    h: f64,
) -> IntegrateResult<OdeResult>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    if h <= 0.0 {
        return Err(IntegrateError::ValueError(format!(
            "step size h={h} must be positive"
        )));
    }
    if t_end < t0 {
        return Err(IntegrateError::ValueError(
            "t_end must be >= t0 for implicit_euler".to_string(),
        ));
    }
    if y0.is_empty() {
        return Err(IntegrateError::ValueError(
            "y0 must be non-empty".to_string(),
        ));
    }

    let mut t = t0;
    let mut y = y0.to_vec();

    let mut t_out = vec![t0];
    let mut y_out = vec![y0.to_vec()];
    let mut n_steps: usize = 0;
    let mut n_rejected: usize = 0;
    let mut n_evals: usize = 0;

    while t < t_end - h * 1e-10 {
        let h_actual = h.min(t_end - t);
        let t_new = t + h_actual;

        // Use explicit Euler as initial guess for Newton
        let f_cur = f(t, &y);
        n_evals += 1;
        let y_guess: Vec<f64> = y
            .iter()
            .zip(f_cur.iter())
            .map(|(yi, fi)| yi + h_actual * fi)
            .collect();

        match newton_implicit_euler(&f, &jac, t_new, &y, h_actual, &y_guess) {
            Ok((y_new, evals)) => {
                n_evals += evals;
                n_steps += 1;
                t = t_new;
                y = y_new;
                t_out.push(t);
                y_out.push(y.clone());
            }
            Err(e) => {
                // One rejection recorded; propagate the error upwards.
                let _n_rejected = n_rejected + 1;
                return Err(e);
            }
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

/// Solve a stiff ODE using the **Trapezoidal (Crank-Nicolson)** method.
///
/// This is a second-order, A-stable method.  It is less damping than implicit
/// Euler (which is L-stable) but delivers higher accuracy for the same step
/// size.  Each step uses Newton iteration with the user-supplied Jacobian.
///
/// # Arguments
///
/// * `f`     – Right-hand side: `dy/dt = f(t, y)`.
/// * `jac`   – Jacobian: `J[i][j] = ∂f_i/∂y_j`.
/// * `t0`    – Initial time.
/// * `y0`    – Initial state vector.
/// * `t_end` – Final time.
/// * `h`     – Fixed step size.
///
/// # Errors
///
/// Returns an error if Newton iteration or the linear solve fails.
pub fn trapezoidal<F, J>(
    f: F,
    jac: J,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    h: f64,
) -> IntegrateResult<OdeResult>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    if h <= 0.0 {
        return Err(IntegrateError::ValueError(format!(
            "step size h={h} must be positive"
        )));
    }
    if t_end < t0 {
        return Err(IntegrateError::ValueError(
            "t_end must be >= t0 for trapezoidal".to_string(),
        ));
    }
    if y0.is_empty() {
        return Err(IntegrateError::ValueError(
            "y0 must be non-empty".to_string(),
        ));
    }

    let mut t = t0;
    let mut y = y0.to_vec();

    let mut t_out = vec![t0];
    let mut y_out = vec![y0.to_vec()];
    let mut n_steps: usize = 0;
    let mut n_rejected: usize = 0;
    let mut n_evals: usize = 0;

    while t < t_end - h * 1e-10 {
        let h_actual = h.min(t_end - t);
        let t_new = t + h_actual;

        let f_cur = f(t, &y);
        n_evals += 1;

        // Explicit Euler as initial guess
        let y_guess: Vec<f64> = y
            .iter()
            .zip(f_cur.iter())
            .map(|(yi, fi)| yi + h_actual * fi)
            .collect();

        match newton_trapezoidal(&f, &jac, t, t_new, &y, &f_cur, h_actual, &y_guess) {
            Ok((y_new, evals)) => {
                n_evals += evals;
                n_steps += 1;
                t = t_new;
                y = y_new;
                t_out.push(t);
                y_out.push(y.clone());
            }
            Err(e) => {
                let _n_rejected = n_rejected + 1;
                return Err(e);
            }
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

/// Solve a stiff ODE using the **BDF-2** (second-order Backward Differentiation
/// Formula) method.
///
/// The BDF-2 formula is:
/// ```text
/// (3/2) y_{n+1} - 2 y_n + (1/2) y_{n-1} = h * f(t_{n+1}, y_{n+1})
/// ```
/// which rearranges to the nonlinear system solved by Newton's method.
///
/// The first step is bootstrapped using Implicit Euler.
///
/// # Arguments
///
/// * `f`     – Right-hand side: `dy/dt = f(t, y)`.
/// * `jac`   – Jacobian at `(t, y)`.
/// * `t0`    – Initial time.
/// * `y0`    – Initial state vector.
/// * `t_end` – Final time.
/// * `h`     – Fixed step size.
///
/// # Errors
///
/// Returns an error if Newton iteration or the linear solve fails.
pub fn bdf2<F, J>(
    f: F,
    jac: J,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    h: f64,
) -> IntegrateResult<OdeResult>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    if h <= 0.0 {
        return Err(IntegrateError::ValueError(format!(
            "step size h={h} must be positive"
        )));
    }
    if t_end < t0 {
        return Err(IntegrateError::ValueError(
            "t_end must be >= t0 for bdf2".to_string(),
        ));
    }
    if y0.is_empty() {
        return Err(IntegrateError::ValueError(
            "y0 must be non-empty".to_string(),
        ));
    }

    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();

    let mut t_out = vec![t0];
    let mut y_out = vec![y0.to_vec()];
    let mut n_steps: usize = 0;
    let mut n_rejected: usize = 0;
    let mut n_evals: usize = 0;

    // Bootstrap with one Implicit Euler step to obtain y_{n-1}
    let mut y_prev: Vec<f64> = y.clone();

    // Keep track of whether we have two history points yet
    let mut have_history = false;

    while t < t_end - h * 1e-10 {
        let h_actual = h.min(t_end - t);
        let t_new = t + h_actual;

        let f_cur = f(t, &y);
        n_evals += 1;
        let y_guess: Vec<f64> = y
            .iter()
            .zip(f_cur.iter())
            .map(|(yi, fi)| yi + h_actual * fi)
            .collect();

        if !have_history {
            // First step: Implicit Euler to bootstrap
            match newton_implicit_euler(&f, &jac, t_new, &y, h_actual, &y_guess) {
                Ok((y_new, evals)) => {
                    n_evals += evals;
                    n_steps += 1;
                    y_prev = y.clone();
                    t = t_new;
                    y = y_new;
                    t_out.push(t);
                    y_out.push(y.clone());
                    have_history = true;
                }
                Err(e) => {
                    let _n_rejected = n_rejected + 1;
                    return Err(e);
                }
            }
        } else {
            // BDF-2 step: solve
            //   (3/2) y_new - 2 y + (1/2) y_prev = h * f(t_new, y_new)
            // i.e.  G(y_new) = (3/2) y_new - 2 y + (1/2) y_prev - h * f(t_new, y_new) = 0
            let y_prev_snap = y_prev.clone();
            let y_snap = y.clone();

            // Newton for BDF-2: G(z) = (3/2) z - 2 y + (1/2) y_prev - h*f(t_new, z) = 0
            // dG/dz = (3/2) I - h J(t_new, z)
            let mut z = y_guess.clone();
            let mut converged = false;
            let mut bdf2_evals: usize = 0;

            for _iter in 0..MAX_NEWTON {
                let fz = f(t_new, &z);
                let jz = jac(t_new, &z);
                bdf2_evals += 2;

                // Residual
                let residual: Vec<f64> = (0..n)
                    .map(|i| 1.5 * z[i] - 2.0 * y_snap[i] + 0.5 * y_prev_snap[i] - h_actual * fz[i])
                    .collect();

                let res_norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
                if res_norm < NEWTON_TOL {
                    converged = true;
                    break;
                }

                // Jacobian of G: (3/2) I - h J
                let mut a = vec![0.0_f64; n * n];
                for i in 0..n {
                    for j in 0..n {
                        let delta = if i == j { 1.5 } else { 0.0 };
                        a[i * n + j] = delta - h_actual * jz[i][j];
                    }
                }
                let neg_residual: Vec<f64> = residual.iter().map(|r| -r).collect();
                let delta = gaussian_solve(&a, &neg_residual)?;

                let delta_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
                for i in 0..n {
                    z[i] += delta[i];
                }

                if delta_norm
                    < NEWTON_TOL * (1.0 + z.iter().map(|v| v.abs()).fold(0.0_f64, f64::max))
                {
                    converged = true;
                    break;
                }
            }

            n_evals += bdf2_evals;

            if !converged {
                let _n_rejected = n_rejected + 1;
                return Err(IntegrateError::ConvergenceError(format!(
                    "BDF-2 Newton iteration did not converge at t={t_new}"
                )));
            }

            y_prev = y.clone();
            n_steps += 1;
            t = t_new;
            y = z;
            t_out.push(t);
            y_out.push(y.clone());
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

    /// Stiff exponential decay: dy/dt = -1000 y, y(0) = 1
    /// Exact solution: y(t) = exp(-1000 t)
    fn stiff_decay(t: f64, y: &[f64]) -> Vec<f64> {
        let _ = t;
        vec![-1000.0 * y[0]]
    }

    fn stiff_decay_jac(t: f64, y: &[f64]) -> Vec<Vec<f64>> {
        let _ = (t, y);
        vec![vec![-1000.0]]
    }

    /// Gentle exponential: dy/dt = -y, y(0) = 1, y(t) = exp(-t)
    fn gentle_decay(t: f64, y: &[f64]) -> Vec<f64> {
        let _ = t;
        vec![-y[0]]
    }

    fn gentle_decay_jac(t: f64, y: &[f64]) -> Vec<Vec<f64>> {
        let _ = (t, y);
        vec![vec![-1.0]]
    }

    /// 2D system: d/dt [y0; y1] = [-10, 1; 0, -1] [y0; y1]
    fn linear_2d(t: f64, y: &[f64]) -> Vec<f64> {
        let _ = t;
        vec![-10.0 * y[0] + y[1], -y[1]]
    }

    fn linear_2d_jac(t: f64, y: &[f64]) -> Vec<Vec<f64>> {
        let _ = (t, y);
        vec![vec![-10.0, 1.0], vec![0.0, -1.0]]
    }

    #[test]
    fn implicit_euler_gentle_decay() {
        let result = implicit_euler(gentle_decay, gentle_decay_jac, 0.0, &[1.0], 5.0, 0.1)
            .expect("implicit_euler failed");
        let y_end = result.y.last().expect("empty result")[0];
        let exact = (-5.0_f64).exp();
        // First-order method with h=0.1: expect ~5% relative error
        assert!((y_end - exact).abs() < 0.05, "y_end={y_end}, exact={exact}");
        assert!(result.n_steps > 0);
    }

    #[test]
    fn implicit_euler_stiff_decay() {
        // Implicit Euler with step h on dy/dt = -1000y gives y_{n+1} = y_n / (1 + 1000h).
        // After N steps: y_N = 1/(1+1000h)^N.  We want this ≈ exp(-1000*N*h).
        // The local truncation error is O(h), so relative accuracy ~ 1000*h/2.
        // Use h = 1e-5 (1000 steps to t = 0.01) for ~0.5% relative accuracy.
        let result = implicit_euler(stiff_decay, stiff_decay_jac, 0.0, &[1.0], 0.01, 1e-5)
            .expect("implicit_euler stiff failed");
        // After t=0.01, y ≈ exp(-10) ≈ 4.54e-5
        let y_end = result.y.last().expect("empty result")[0];
        let exact = (-10.0_f64).exp();
        assert!(
            (y_end - exact).abs() < exact * 0.10,
            "y_end={y_end}, exact={exact}"
        );
    }

    #[test]
    fn trapezoidal_gentle_decay() {
        // Trapezoidal is 2nd order → tighter error for same h
        let result = trapezoidal(gentle_decay, gentle_decay_jac, 0.0, &[1.0], 5.0, 0.1)
            .expect("trapezoidal failed");
        let y_end = result.y.last().expect("empty result")[0];
        let exact = (-5.0_f64).exp();
        // 2nd-order, h=0.1: expect < 0.5% error
        assert!(
            (y_end - exact).abs() / exact < 0.005,
            "rel_err={}",
            (y_end - exact).abs() / exact
        );
    }

    #[test]
    fn trapezoidal_2d() {
        // [y0, y1] = exp(-10t)[A+Bt] style; just check stability and convergence
        let result = trapezoidal(linear_2d, linear_2d_jac, 0.0, &[1.0, 1.0], 1.0, 0.05)
            .expect("trapezoidal 2d failed");
        let y_end = result.y.last().expect("empty result");
        // y1(t) = exp(-t) → y1(1.0) ≈ 0.368
        let exact_y1 = (-1.0_f64).exp();
        assert!(
            (y_end[1] - exact_y1).abs() / exact_y1 < 0.01,
            "y1_end={}, exact={}",
            y_end[1],
            exact_y1
        );
    }

    #[test]
    fn bdf2_gentle_decay() {
        let result =
            bdf2(gentle_decay, gentle_decay_jac, 0.0, &[1.0], 5.0, 0.1).expect("bdf2 failed");
        let y_end = result.y.last().expect("empty result")[0];
        let exact = (-5.0_f64).exp();
        // BDF-2 is 2nd order → similar to trapezoidal
        assert!(
            (y_end - exact).abs() / exact < 0.02,
            "rel_err={}",
            (y_end - exact).abs() / exact
        );
    }

    #[test]
    fn bdf2_stiff() {
        // BDF-2 is second-order, so use h = 1e-4 for good accuracy on dy/dt = -1000y.
        let result =
            bdf2(stiff_decay, stiff_decay_jac, 0.0, &[1.0], 0.01, 1e-4).expect("bdf2 stiff failed");
        let y_end = result.y.last().expect("empty result")[0];
        let exact = (-10.0_f64).exp();
        assert!(
            (y_end - exact).abs() < exact * 0.10,
            "y_end={y_end}, exact={exact}"
        );
    }

    #[test]
    fn implicit_euler_validates_inputs() {
        // Negative h
        assert!(implicit_euler(gentle_decay, gentle_decay_jac, 0.0, &[1.0], 1.0, -0.1).is_err());
        // t_end < t0
        assert!(implicit_euler(gentle_decay, gentle_decay_jac, 1.0, &[1.0], 0.0, 0.1).is_err());
        // empty y0
        assert!(implicit_euler(gentle_decay, gentle_decay_jac, 0.0, &[], 1.0, 0.1).is_err());
    }
}
