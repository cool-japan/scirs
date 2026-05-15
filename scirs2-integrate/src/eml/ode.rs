//! BDF1 (implicit Euler) stiff ODE solver with exact symbolic Jacobian.
//!
//! # Variable convention
//!
//! - `Var(0)` — time `t`
//! - `Var(1)` through `Var(n)` — state components `y[0]` through `y[n-1]`
//!
//! The evaluation context passed to each RHS component is therefore
//! `[t, y[0], y[1], …, y[n-1]]` (length `n + 1`).
//!
//! # Algorithm
//!
//! BDF1 (backward Euler) requires at each step the solution of
//!
//! ```text
//! F(y_next) = y_next − y_curr − h · f(t_next, y_next) = 0
//! ```
//!
//! which is solved by Newton iterations with the exact Jacobian
//! `J_F = I − h · J_f` where `J_f[i][j] = ∂f_i/∂y_j` is precomputed
//! symbolically once before the time loop.

use std::sync::Arc;

use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::{grad, LoweredOp};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Summary of a successful [`solve_ivp_symbolic`] run.
#[derive(Debug, Clone)]
pub struct SymbolicOdeResult {
    /// Time points (including `t_span[0]`).
    pub t: Vec<f64>,
    /// State vectors at each time point (same length as `t`).
    pub y: Vec<Array1<f64>>,
    /// Total accepted steps.
    pub n_steps: usize,
    /// Total Newton iterations (summed over all accepted and rejected steps).
    pub n_newton: usize,
    /// Number of symbolic Jacobian matrix evaluations.
    pub n_jac_evals: usize,
}

/// Errors that can occur during [`solve_ivp_symbolic`].
#[derive(Debug)]
pub enum SymbolicOdeError {
    /// The symbolic evaluator returned an error (domain violation, unbound
    /// variable, division by zero, etc.).
    EvalError(String),
    /// The BDF1 Jacobian `(I − h·J)` became singular and step-halving
    /// exhausted the minimum step size.
    SingularJacobian,
    /// Newton's method did not converge within the allotted iterations and
    /// step-halving exhausted the minimum step size.
    NewtonNotConverged {
        /// The step index at which convergence failed.
        step: usize,
    },
    /// `rhs.len()` does not match `y0.len()`.
    DimMismatch {
        /// Number of RHS components supplied.
        rhs_len: usize,
        /// Length of the initial-condition vector.
        y0_len: usize,
    },
    /// A step was rejected because the step size fell below the minimum after
    /// repeated halving (singular Jacobian during a non-initial step).
    StepRejected {
        /// The step index that was rejected.
        step: usize,
    },
    /// Invalid input argument.
    InvalidInput(String),
}

impl std::fmt::Display for SymbolicOdeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolicOdeError::EvalError(msg) => write!(f, "symbolic eval error: {msg}"),
            SymbolicOdeError::SingularJacobian => {
                write!(f, "BDF1 Jacobian (I−h·J) is singular after step halving")
            }
            SymbolicOdeError::NewtonNotConverged { step } => {
                write!(
                    f,
                    "Newton iterations did not converge at step {step} after step halving"
                )
            }
            SymbolicOdeError::DimMismatch { rhs_len, y0_len } => write!(
                f,
                "dimension mismatch: rhs has {rhs_len} components but y0 has {y0_len} entries"
            ),
            SymbolicOdeError::StepRejected { step } => {
                write!(
                    f,
                    "step {step} rejected: step size fell below minimum after repeated halving"
                )
            }
            SymbolicOdeError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for SymbolicOdeError {}

// ---------------------------------------------------------------------------
// Main solver
// ---------------------------------------------------------------------------

/// BDF1 (implicit Euler) stiff ODE solver with exact symbolic Jacobian.
///
/// Solves the initial value problem
///
/// ```text
/// dy/dt = f(t, y),  y(t_span[0]) = y0
/// ```
///
/// where each component of `f` is given as a [`LoweredOp`] expression wrapped
/// in [`Arc`].
///
/// # Variable convention
///
/// - `Var(0)` — time `t`
/// - `Var(1)` through `Var(n)` — state components `y[0]` through `y[n-1]`
///
/// # Arguments
///
/// - `rhs` — one `LoweredOp` per state component (length `n`)
/// - `t_span` — `[t_start, t_end]` with `t_start < t_end`
/// - `y0` — initial state vector (length `n`)
/// - `h0` — initial step size (must be `> 0`)
/// - `rtol` — relative tolerance for Newton convergence
/// - `atol` — absolute tolerance (added to denominator to prevent division
///   by zero when state is zero; must be `> 0`)
/// - `max_steps` — maximum number of accepted time steps
///
/// # Errors
///
/// - [`SymbolicOdeError::DimMismatch`] when `rhs.len() != y0.len()`
/// - [`SymbolicOdeError::InvalidInput`] when `h0 <= 0`, `atol <= 0`,
///   `rtol <= 0`, or `t_span[0] >= t_span[1]`
/// - [`SymbolicOdeError::EvalError`] when symbolic evaluation fails
/// - [`SymbolicOdeError::NewtonNotConverged`] / [`SymbolicOdeError::StepRejected`]
///   when the Newton solver cannot converge even after step halving
pub fn solve_ivp_symbolic(
    rhs: &[Arc<LoweredOp>],
    t_span: [f64; 2],
    y0: ArrayView1<f64>,
    h0: f64,
    rtol: f64,
    atol: f64,
    max_steps: usize,
) -> Result<SymbolicOdeResult, SymbolicOdeError> {
    // --- Input validation ---------------------------------------------------
    let n = rhs.len();
    if n != y0.len() {
        return Err(SymbolicOdeError::DimMismatch {
            rhs_len: n,
            y0_len: y0.len(),
        });
    }
    if h0 <= 0.0 {
        return Err(SymbolicOdeError::InvalidInput(format!(
            "h0 must be positive, got {h0}"
        )));
    }
    if atol <= 0.0 {
        return Err(SymbolicOdeError::InvalidInput(format!(
            "atol must be positive, got {atol}"
        )));
    }
    if rtol <= 0.0 {
        return Err(SymbolicOdeError::InvalidInput(format!(
            "rtol must be positive, got {rtol}"
        )));
    }
    if t_span[0] >= t_span[1] {
        return Err(SymbolicOdeError::InvalidInput(format!(
            "t_span[0] ({}) must be < t_span[1] ({})",
            t_span[0], t_span[1]
        )));
    }

    // --- Precompute symbolic Jacobian J[i][j] = ∂rhs_i/∂y_j ----------------
    // Variables: Var(0) = t, Var(j+1) = y[j]
    let jac_op: Vec<Vec<LoweredOp>> = rhs
        .iter()
        .map(|fi| (0..n).map(|j| grad(fi.as_ref(), j + 1)).collect::<Vec<_>>())
        .collect();

    // --- Time integration loop ----------------------------------------------
    const MAX_NEWTON: usize = 10;
    const H_MIN: f64 = 1e-12;
    const MAX_HALVINGS: usize = 20;

    let mut t_curr = t_span[0];
    let mut y_curr = y0.to_owned();
    let mut h = h0;

    let mut t_out: Vec<f64> = Vec::new();
    let mut y_out: Vec<Array1<f64>> = Vec::new();
    let mut n_steps = 0usize;
    let mut n_newton = 0usize;
    let mut n_jac_evals = 0usize;

    // Push initial point
    t_out.push(t_curr);
    y_out.push(y_curr.clone());

    // Shared binding buffer: [t, y[0], ..., y[n-1]]
    let mut bindings = vec![0.0_f64; n + 1];

    'outer: for step_idx in 0..max_steps {
        if t_curr >= t_span[1] {
            break;
        }
        // Clamp to endpoint
        let t_next = (t_curr + h).min(t_span[1]);
        let h_eff = t_next - t_curr;

        // Newton iteration with step-halving on failure
        let mut h_try = h_eff;
        let mut t_try = t_curr + h_try;
        let mut halvings = 0usize;

        let accepted = loop {
            // Initial guess: forward Euler
            let mut y_next = y_curr.clone();

            // Newton inner loop
            let mut newton_ok = false;
            let mut jac_ok = true;

            for newton_iter in 0..MAX_NEWTON {
                n_newton += 1;

                // Build binding slice: [t_try, y_next[0], ..., y_next[n-1]]
                bindings[0] = t_try;
                for k in 0..n {
                    bindings[k + 1] = y_next[k];
                }
                let ctx = EvalCtx::new(&bindings);

                // Evaluate RHS: f(t_try, y_next)
                let mut f_vec = vec![0.0_f64; n];
                for i in 0..n {
                    f_vec[i] = eval_real(rhs[i].as_ref(), &ctx)
                        .map_err(|e| SymbolicOdeError::EvalError(e.to_string()))?;
                }

                // Residual: F = y_next - y_curr - h * f
                let residual: Vec<f64> = (0..n)
                    .map(|i| y_next[i] - y_curr[i] - h_try * f_vec[i])
                    .collect();

                // Evaluate Jacobian J[i][j] = ∂f_i/∂y_j
                n_jac_evals += 1;
                let mut j_num: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        j_num[i][j] = eval_real(&jac_op[i][j], &ctx)
                            .map_err(|e| SymbolicOdeError::EvalError(e.to_string()))?;
                    }
                }

                // Build (I - h·J): lhs[i][j] = δ_{ij} - h_try * J[i][j]
                let lhs: Vec<Vec<f64>> = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| {
                                let delta = if i == j { 1.0 } else { 0.0 };
                                delta - h_try * j_num[i][j]
                            })
                            .collect()
                    })
                    .collect();

                // Solve (I - h·J) · Δy = -F
                let rhs_lin: Vec<f64> = residual.iter().map(|r| -r).collect();
                let delta_y = match solve_linear(&lhs, &rhs_lin) {
                    Some(dy) => dy,
                    None => {
                        // Singular Jacobian — try halving
                        jac_ok = false;
                        break;
                    }
                };

                // Update state
                for k in 0..n {
                    y_next[k] += delta_y[k];
                }

                // Convergence: ||Δy||₂ / (||y_next||₂ + atol) < rtol
                let delta_norm_sq: f64 = delta_y.iter().map(|d| d * d).sum();
                let y_norm_sq: f64 = y_next.iter().map(|v| v * v).sum();
                let denom = y_norm_sq.sqrt() + atol;
                if delta_norm_sq.sqrt() / denom < rtol || newton_iter == MAX_NEWTON - 1 {
                    // Converged on this iteration; accept if the norm check passed
                    if delta_norm_sq.sqrt() / denom < rtol {
                        newton_ok = true;
                    }
                    break;
                }
            }

            if newton_ok && jac_ok {
                // Step accepted
                break (Some((t_try, y_next, h_try)));
            }

            // Step rejected — halve
            halvings += 1;
            if halvings > MAX_HALVINGS || h_try / 2.0 < H_MIN {
                break None;
            }
            h_try /= 2.0;
            t_try = t_curr + h_try;
        };

        match accepted {
            None => {
                // Could not find a working step size
                if n_steps == 0 {
                    return Err(SymbolicOdeError::SingularJacobian);
                }
                return Err(SymbolicOdeError::NewtonNotConverged { step: step_idx });
            }
            Some((t_accepted, y_accepted, h_accepted)) => {
                t_curr = t_accepted;
                y_curr = y_accepted;
                h = h_accepted; // Keep the step size that worked

                t_out.push(t_curr);
                y_out.push(y_curr.clone());
                n_steps += 1;

                if t_curr >= t_span[1] {
                    break 'outer;
                }
            }
        }
    }

    Ok(SymbolicOdeResult {
        t: t_out,
        y: y_out,
        n_steps,
        n_newton,
        n_jac_evals,
    })
}

// ---------------------------------------------------------------------------
// Linear solver helper
// ---------------------------------------------------------------------------

/// Solve `A · x = b` via partial-pivoting Gaussian elimination.
///
/// Returns `Some(x)` on success, or `None` if the matrix is singular
/// (maximum pivot magnitude `< 1e-12`).
///
/// Adapted from `scirs2_optimize::symbolic::solve_linear` (same algorithm,
/// different error convention — returns `Option` so the ODE driver can do
/// step-halving rather than immediately returning an error).
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Some(Vec::new());
    }

    // Build augmented matrix [A | b]
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
        // Find pivot row
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
            return None; // Singular
        }
        mat.swap(k, max_idx);

        // Eliminate below pivot
        for i in (k + 1)..n {
            let factor = mat[i][k] / mat[k][k];
            for j in k..=n {
                let pivot_val = mat[k][j];
                mat[i][j] -= factor * pivot_val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = mat[i][n];
        for j in (i + 1)..n {
            sum -= mat[i][j] * x[j];
        }
        x[i] = sum / mat[i][i];
    }

    Some(x)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    fn make_decay_rhs() -> Vec<Arc<LoweredOp>> {
        // x' = -1000·x  (stiff decay); Var(0)=t, Var(1)=x
        vec![Arc::new(LoweredOp::Mul(
            Box::new(LoweredOp::Const(-1000.0)),
            Box::new(LoweredOp::Var(1)),
        ))]
    }

    #[test]
    fn solve_linear_2x2() {
        // [2 1; 1 3] · x = [5; 10]  → x ≈ [1, 3]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 10.0];
        let x = solve_linear(&a, &b).expect("should solve");
        assert!((x[0] - 1.0).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-10, "x[1] = {}", x[1]);
    }

    #[test]
    fn solve_linear_singular_returns_none() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]]; // rank-1
        let b = vec![3.0, 6.0];
        assert!(solve_linear(&a, &b).is_none());
    }

    #[test]
    fn bdf1_dim_mismatch() {
        let rhs = make_decay_rhs();
        // y0 has 2 components but rhs has 1
        let err = solve_ivp_symbolic(
            &rhs,
            [0.0, 0.001],
            arr1(&[1.0, 0.0]).view(),
            1e-4,
            1e-6,
            1e-8,
            200,
        );
        assert!(matches!(err, Err(SymbolicOdeError::DimMismatch { .. })));
    }

    #[test]
    fn bdf1_invalid_h0() {
        let rhs = make_decay_rhs();
        let err = solve_ivp_symbolic(
            &rhs,
            [0.0, 0.001],
            arr1(&[1.0]).view(),
            -0.1,
            1e-6,
            1e-8,
            200,
        );
        assert!(matches!(err, Err(SymbolicOdeError::InvalidInput(_))));
    }

    #[test]
    fn bdf1_stiff_decay_completes() {
        let rhs = make_decay_rhs();
        let result = solve_ivp_symbolic(
            &rhs,
            [0.0, 0.001],
            arr1(&[1.0]).view(),
            1e-4,
            1e-4,
            1e-8,
            500,
        );
        assert!(result.is_ok(), "solver failed: {:?}", result.err());
        let res = result.expect("solver ok");
        assert!(!res.t.is_empty());
        assert!(!res.y.is_empty());
    }
}
