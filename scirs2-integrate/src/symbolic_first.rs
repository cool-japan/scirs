//! Symbolic-first ODE solver: attempts `cas::solve_ode` before falling back
//! to a fixed-step RK4 numerical solver.
//!
//! # Design
//!
//! The dispatcher [`solve_ode_symbolic_or_numerical`] accepts an optional
//! `LoweredOp` representing the symbolic RHS `dx/dt = rhs(t, x)`.  When the
//! `symbolic` feature is enabled and `SolvePreference::SymbolicFirst` is
//! chosen (the default), the function forwards to
//! `scirs2_symbolic::cas::solve_ode`.  On success it returns
//! [`SymbolicOrNumericalResult::Symbolic`]; on any `SolveOdeError` it silently
//! falls back to the supplied numeric closure.
//!
//! When the `symbolic` feature is absent (or `SolvePreference::ForceNumerical`
//! is set) the symbolic path is skipped entirely.
//!
//! # No `unwrap()` policy
//!
//! All fallible operations return `Result`; `rk4_fixed` uses `?`-propagation
//! through `SymbolicFirstError::NumericalFailed`.

use scirs2_core::ndarray::{Array1, Array2};

#[cfg(feature = "symbolic")]
use scirs2_symbolic::{
    cas::{solve_ode, OdeKind, OdeSolution, SolveOdeError},
    eml::{eval_real, EvalCtx, LoweredOp},
};

// ---------------------------------------------------------------------------
// Public types (always present, regardless of feature)
// ---------------------------------------------------------------------------

/// Preference for solver selection.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SolvePreference {
    /// Try symbolic first, fall back to numerical on failure.
    #[default]
    SymbolicFirst,
    /// Skip symbolic attempt entirely.
    ForceNumerical,
}

/// Options for [`solve_ode_symbolic_or_numerical`].
#[derive(Debug, Clone)]
pub struct OdeOpts {
    /// Maximum number of steps for the numerical fallback.
    pub max_steps: usize,
    /// Relative tolerance (reserved for future adaptive step-size control).
    pub rtol: f64,
    /// Absolute tolerance (reserved for future adaptive step-size control).
    pub atol: f64,
    /// Whether to attempt symbolic solution first.
    pub preferred: SolvePreference,
}

impl Default for OdeOpts {
    fn default() -> Self {
        OdeOpts {
            max_steps: 10_000,
            rtol: 1e-6,
            atol: 1e-8,
            preferred: SolvePreference::SymbolicFirst,
        }
    }
}

/// Outcome of [`solve_ode_symbolic_or_numerical`].
#[cfg(feature = "symbolic")]
pub enum SymbolicOrNumericalResult {
    /// Closed-form solution found by `cas::solve_ode`.
    Symbolic {
        /// Symbolic expression for `x(t)`.
        x_of_t: LoweredOp,
        /// ODE family that was solved.
        kind: OdeKind,
        /// Var ids of free integration constants still present in `x_of_t`.
        /// Empty when an IC was supplied and resolved the constant.
        integration_constants: Vec<usize>,
    },
    /// Numerical trajectory from the RK4 fallback.
    Numerical {
        /// Columns: `[t, x]` — shape `(n_steps + 1, 2)`.
        trajectory: Array2<f64>,
        /// Time vector, length `n_steps + 1`.
        time: Array1<f64>,
    },
}

/// Outcome of [`solve_ode_symbolic_or_numerical`] (numerical-only build).
#[cfg(not(feature = "symbolic"))]
pub enum SymbolicOrNumericalResult {
    /// Numerical trajectory from the RK4 fallback.
    Numerical {
        /// Columns: `[t, x]` — shape `(n_steps + 1, 2)`.
        trajectory: Array2<f64>,
        /// Time vector, length `n_steps + 1`.
        time: Array1<f64>,
    },
}

/// Error type for this module.
#[derive(Debug, Clone)]
pub enum SymbolicFirstError {
    /// The time interval is invalid (`t_end` must be strictly greater than `t0`).
    InvalidInterval,
    /// The numerical solver failed to produce a valid trajectory.
    NumericalFailed(String),
}

impl std::fmt::Display for SymbolicFirstError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInterval => {
                write!(f, "invalid ODE interval: t_end must be > t0")
            }
            Self::NumericalFailed(msg) => {
                write!(f, "numerical solver failed: {msg}")
            }
        }
    }
}

impl std::error::Error for SymbolicFirstError {}

// ---------------------------------------------------------------------------
// Main dispatcher — symbolic feature enabled
// ---------------------------------------------------------------------------

/// Solve `dx/dt = rhs(t, x)` symbolically if possible, otherwise numerically.
///
/// # Parameters
///
/// - `rhs_symbolic`: optional [`LoweredOp`] for the RHS in terms of
///   `Var(x_var)` and `Var(t_var)`.  Ignored when `None` or when
///   `preferred == ForceNumerical`.
/// - `rhs_numeric`: numeric closure `f(t, x) -> f64` used as the RK4
///   fallback (or always, when symbolic path is skipped/fails).
/// - `x_var`: `Var` id for the dependent variable `x`.
/// - `t_var`: `Var` id for the independent variable `t`.
/// - `ic`: initial condition `(t0, x0)`.
/// - `t_end`: end time; must be strictly greater than `t0`.
/// - `opts`: solver options.
///
/// # Returns
///
/// [`SymbolicOrNumericalResult::Symbolic`] when `cas::solve_ode` succeeds
/// and `preferred == SymbolicFirst`; otherwise
/// [`SymbolicOrNumericalResult::Numerical`] from the fixed-step RK4.
#[cfg(feature = "symbolic")]
pub fn solve_ode_symbolic_or_numerical<F>(
    rhs_symbolic: Option<&LoweredOp>,
    rhs_numeric: F,
    x_var: usize,
    t_var: usize,
    ic: (f64, f64),
    t_end: f64,
    opts: &OdeOpts,
) -> Result<SymbolicOrNumericalResult, SymbolicFirstError>
where
    F: Fn(f64, f64) -> f64,
{
    let (t0, _x0) = ic;
    if t_end <= t0 {
        return Err(SymbolicFirstError::InvalidInterval);
    }

    // Attempt symbolic path if caller has not opted out.
    if opts.preferred == SolvePreference::SymbolicFirst {
        if let Some(sym_rhs) = rhs_symbolic {
            match solve_ode(sym_rhs, x_var, t_var, Some(ic)) {
                Ok(OdeSolution {
                    x_of_t,
                    kind,
                    integration_constants,
                }) => {
                    return Ok(SymbolicOrNumericalResult::Symbolic {
                        x_of_t,
                        kind,
                        integration_constants,
                    });
                }
                Err(_e) => {
                    // Any SolveOdeError → fall through to numerical.
                }
            }
        }
    }

    // Numerical fallback: fixed-step RK4.
    let (trajectory, time) = rk4_fixed(&rhs_numeric, t0, ic.1, t_end, opts.max_steps)?;
    Ok(SymbolicOrNumericalResult::Numerical { trajectory, time })
}

// ---------------------------------------------------------------------------
// Main dispatcher — symbolic feature disabled
// ---------------------------------------------------------------------------

/// Solve `dx/dt = rhs(t, x)` numerically (symbolic feature not available).
///
/// Signature intentionally differs from the `symbolic`-enabled overload:
/// the `rhs_symbolic`, `x_var`, and `t_var` parameters are absent because
/// there is nothing to do with them.
#[cfg(not(feature = "symbolic"))]
pub fn solve_ode_symbolic_or_numerical<F>(
    rhs_numeric: F,
    ic: (f64, f64),
    t_end: f64,
    opts: &OdeOpts,
) -> Result<SymbolicOrNumericalResult, SymbolicFirstError>
where
    F: Fn(f64, f64) -> f64,
{
    let (t0, x0) = ic;
    if t_end <= t0 {
        return Err(SymbolicFirstError::InvalidInterval);
    }
    let (trajectory, time) = rk4_fixed(&rhs_numeric, t0, x0, t_end, opts.max_steps)?;
    Ok(SymbolicOrNumericalResult::Numerical { trajectory, time })
}

// ---------------------------------------------------------------------------
// rhs_from_symbolic_only — derive numeric closure from LoweredOp
// ---------------------------------------------------------------------------

/// Build a numeric RHS closure `f(t, x) -> f64` by evaluating a `LoweredOp`
/// JIT-style using the `EvalCtx` positional evaluator.
///
/// `x_var` and `t_var` are the `Var` ids used inside `sym_rhs`.
/// The closure allocates a small `Vec<f64>` per call (size `x_var.max(t_var) + 1`)
/// so that positional indexing works correctly regardless of which id is larger.
///
/// # NaN policy
///
/// Domain errors from `eval_real` (e.g. `ln` of a negative) are surfaced as
/// `f64::NAN`.  The RK4 fallback will propagate the NaN; callers that need
/// strict error handling should validate the trajectory afterwards.
/// Safety: domain errors flow through as NaN; the numerical solver detects
/// divergence via NaN propagation.
#[cfg(feature = "symbolic")]
pub fn rhs_from_symbolic_only(
    sym_rhs: LoweredOp,
    x_var: usize,
    t_var: usize,
) -> impl Fn(f64, f64) -> f64 {
    let capacity = x_var.max(t_var) + 1;
    move |t: f64, x: f64| {
        let mut bindings = vec![0.0_f64; capacity];
        // Safety: both indices are < capacity by construction.
        bindings[t_var] = t;
        bindings[x_var] = x;
        let ctx = EvalCtx::new(&bindings);
        // Safety: domain errors flow through as NaN; the numerical solver
        // handles divergence via NaN propagation — no panic.
        eval_real(&sym_rhs, &ctx).unwrap_or(f64::NAN)
    }
}

// ---------------------------------------------------------------------------
// Fixed-step RK4
// ---------------------------------------------------------------------------

/// Simple fixed-step RK4 integrator.
///
/// Returns `(trajectory, time)` where:
/// - `trajectory` has shape `(n_steps + 1, 2)` with columns `[t, x]`.
/// - `time` has length `n_steps + 1`.
fn rk4_fixed<F>(
    f: &F,
    t0: f64,
    x0: f64,
    t_end: f64,
    n_steps: usize,
) -> Result<(Array2<f64>, Array1<f64>), SymbolicFirstError>
where
    F: Fn(f64, f64) -> f64,
{
    let h = (t_end - t0) / (n_steps as f64);
    let mut t = t0;
    let mut x = x0;

    let capacity = n_steps + 1;
    let mut flat: Vec<f64> = Vec::with_capacity(capacity * 2);
    let mut time_vec: Vec<f64> = Vec::with_capacity(capacity);

    flat.push(t);
    flat.push(x);
    time_vec.push(t);

    for _ in 0..n_steps {
        let k1 = f(t, x);
        let k2 = f(t + h / 2.0, x + h * k1 / 2.0);
        let k3 = f(t + h / 2.0, x + h * k2 / 2.0);
        let k4 = f(t + h, x + h * k3);
        x += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        t += h;
        flat.push(t);
        flat.push(x);
        time_vec.push(t);
    }

    let n = time_vec.len();
    let trajectory = Array2::from_shape_vec((n, 2), flat)
        .map_err(|e| SymbolicFirstError::NumericalFailed(e.to_string()))?;
    let time = Array1::from_vec(time_vec);
    Ok((trajectory, time))
}
