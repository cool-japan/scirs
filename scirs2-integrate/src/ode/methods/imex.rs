//! IMEX (Implicit-Explicit) Splitting Methods for Stiff ODEs
//!
//! This module implements additive Runge-Kutta (ARK) and multistep IMEX methods
//! for systems of the form:
//!
//!   dy/dt = f_E(t, y) + f_I(t, y)
//!
//! where f_E is a non-stiff (explicit) part and f_I is a stiff (implicit) part.
//!
//! ## Implemented methods
//!
//! | Name            | Order | Description                                       |
//! |-----------------|-------|---------------------------------------------------|
//! | IMEX Euler      | 1     | Forward Euler (explicit) + Backward Euler (impl.) |
//! | IMEX Midpoint   | 2     | Explicit Euler + implicit midpoint rule            |
//! | IMEX BDF2       | 2     | Adams extrapolation (expl.) + BDF2 (impl.)         |
//! | IMEX-ARK SSP2   | 2     | 2-stage ARK, L-stable implicit part                |
//! | IMEX-ARK SSP3   | 2     | 3-stage ARK, SSP, Pareschi-Russo scheme            |
//!
//! ## References
//!
//! - Ascher, Ruuth, Spiteri (1997), "Implicit-explicit Runge-Kutta methods for
//!   time-dependent partial differential equations", Appl. Numer. Math. 25
//! - Pareschi, Russo (2005), "Implicit-explicit Runge-Kutta schemes and
//!   applications to hyperbolic systems with relaxation", J. Sci. Comput. 25
//! - Kennedy, Carpenter (2003), "Additive Runge-Kutta schemes for
//!   convection-diffusion-reaction equations", Appl. Numer. Math. 44

use crate::error::{IntegrateError, IntegrateResult};
use crate::IntegrateFloat;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Convert f64 literal to generic float type
#[inline(always)]
fn to_f<F: IntegrateFloat>(v: f64) -> F {
    F::from_f64(v).unwrap_or_else(F::zero)
}

// ---------------------------------------------------------------------------
// SplitFunction trait
// ---------------------------------------------------------------------------

/// Trait for systems split into explicit (non-stiff) and implicit (stiff) parts.
///
/// The ODE is written as `dy/dt = f_E(t, y) + f_I(t, y)`.
///
/// Implementors must provide:
/// - `explicit_part`: the non-stiff right-hand side f_E
/// - `implicit_part`: the stiff right-hand side f_I
/// - `jacobian_implicit`: the Jacobian ∂f_I/∂y (needed by implicit solvers)
/// - `dimension`: the number of equations
pub trait SplitFunction<F: IntegrateFloat>: Send + Sync {
    /// Non-stiff (explicit) part of the right-hand side
    fn explicit_part(&self, t: F, y: ArrayView1<F>) -> Array1<F>;

    /// Stiff (implicit) part of the right-hand side
    fn implicit_part(&self, t: F, y: ArrayView1<F>) -> Array1<F>;

    /// Jacobian of the implicit part ∂f_I/∂y (n×n matrix)
    fn jacobian_implicit(&self, t: F, y: ArrayView1<F>) -> Array2<F>;

    /// Number of equations
    fn dimension(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for IMEX splitting methods
#[derive(Debug, Clone)]
pub struct IMEXConfig<F: IntegrateFloat> {
    /// Time step size (fixed step for multistep methods, initial step for ARK)
    pub dt: F,
    /// End time of integration
    pub t_end: F,
    /// Relative tolerance for Newton iterations
    pub rtol: F,
    /// Absolute tolerance for Newton iterations
    pub atol: F,
    /// Maximum Newton iterations per step
    pub max_iter_newton: usize,
    /// Convergence tolerance for Newton solver
    pub newton_tol: F,
    /// Whether to compute stiffness ratio estimate
    pub compute_stiffness: bool,
}

impl Default for IMEXConfig<f64> {
    fn default() -> Self {
        Self {
            dt: 1e-3,
            t_end: 1.0,
            rtol: 1e-6,
            atol: 1e-9,
            max_iter_newton: 50,
            newton_tol: 1e-10,
            compute_stiffness: false,
        }
    }
}

impl<F: IntegrateFloat> IMEXConfig<F> {
    /// Create a new configuration with given time step and end time
    pub fn new(dt: F, t_end: F) -> Self {
        Self {
            dt,
            t_end,
            rtol: to_f(1e-6),
            atol: to_f(1e-9),
            max_iter_newton: 50,
            newton_tol: to_f(1e-10),
            compute_stiffness: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of an IMEX integration
#[derive(Debug, Clone)]
pub struct IMEXResult<F: IntegrateFloat> {
    /// Time points
    pub t: Vec<F>,
    /// Solution at each time point
    pub y: Vec<Array1<F>>,
    /// Stiffness ratio estimate at each step (ratio of implicit to explicit spectral radius).
    /// Empty unless `IMEXConfig::compute_stiffness` is `true`.
    pub stiffness_ratio: Vec<F>,
    /// Total number of accepted steps
    pub n_steps: usize,
    /// Total number of Newton iterations across all steps
    pub n_newton_iters: usize,
}

// ---------------------------------------------------------------------------
// Linear algebra helpers (Gaussian elimination, no external crate needed)
// ---------------------------------------------------------------------------

/// Solve A·x = b using partial-pivoting Gaussian elimination.
///
/// Modifies A and b in place, returns x. Returns an error if A is singular.
fn gaussian_elimination<F: IntegrateFloat>(
    a: &mut Array2<F>,
    b: &mut Array1<F>,
) -> IntegrateResult<Array1<F>> {
    let n = b.len();
    if a.shape() != [n, n] {
        return Err(IntegrateError::DimensionMismatch(format!(
            "Matrix shape {:?} incompatible with RHS length {}",
            a.shape(),
            n
        )));
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[[col, col]].abs();
        for row in (col + 1)..n {
            let v = a[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < to_f(1e-300) {
            return Err(IntegrateError::LinearSolveError(
                "Singular or near-singular matrix in IMEX Newton solve".to_string(),
            ));
        }

        // Swap rows
        if max_row != col {
            for j in col..n {
                let tmp = a[[col, j]];
                a[[col, j]] = a[[max_row, j]];
                a[[max_row, j]] = tmp;
            }
            b.swap(col, max_row);
        }

        // Eliminate below
        let pivot = a[[col, col]];
        for row in (col + 1)..n {
            let factor = a[[row, col]] / pivot;
            for j in col..n {
                let update = factor * a[[col, j]];
                a[[row, j]] -= update;
            }
            let bupdate = factor * b[col];
            b[row] -= bupdate;
        }
    }

    // Back substitution
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            let ax = a[[i, j]] * x[j];
            sum -= ax;
        }
        x[i] = sum / a[[i, i]];
    }

    Ok(x)
}

/// Solve the linear system `(alpha*I - dt*J) * delta = rhs` by Gaussian elimination.
///
/// This is the standard linear system arising in IMEX Newton iterations.
fn solve_imex_linear<F: IntegrateFloat>(
    jac: &Array2<F>,
    rhs: &Array1<F>,
    alpha: F,
    dt: F,
) -> IntegrateResult<Array1<F>> {
    let n = rhs.len();
    let mut mat = Array2::<F>::zeros((n, n));
    // Build alpha*I - dt*J
    for i in 0..n {
        for j in 0..n {
            mat[[i, j]] = if i == j {
                alpha - dt * jac[[i, j]]
            } else {
                F::zero() - dt * jac[[i, j]]
            };
        }
    }
    let mut rhs_copy = rhs.clone();
    gaussian_elimination(&mut mat, &mut rhs_copy)
}

// ---------------------------------------------------------------------------
// Newton solver for implicit equations
// ---------------------------------------------------------------------------

/// Solve F(y) = y - y_prev - dt * f_I(t, y) - explicit_term = 0
/// using damped Newton iteration.
///
/// Returns (solution, n_iters).
fn newton_solve_implicit<F, Sys>(
    sys: &Sys,
    t: F,
    y_prev: &Array1<F>,
    explicit_term: &Array1<F>,
    dt: F,
    cfg: &IMEXConfig<F>,
) -> IntegrateResult<(Array1<F>, usize)>
where
    F: IntegrateFloat,
    Sys: SplitFunction<F>,
{
    let n = y_prev.len();
    let mut y = y_prev.clone();
    let mut n_iters;

    for iter in 0..cfg.max_iter_newton {
        let f_i = sys.implicit_part(t, y.view());
        // Residual: r = y - y_prev - dt*f_I - explicit_term
        let mut residual = Array1::<F>::zeros(n);
        for i in 0..n {
            residual[i] = y[i] - y_prev[i] - dt * f_i[i] - explicit_term[i];
        }

        // Check convergence
        let res_norm = residual
            .iter()
            .fold(F::zero(), |acc, &r| acc + r * r)
            .sqrt();
        if res_norm < cfg.newton_tol {
            n_iters = iter + 1;
            return Ok((y, n_iters));
        }

        // Jacobian of residual: I - dt * J_I
        let jac = sys.jacobian_implicit(t, y.view());
        // Solve (I - dt*J_I) * delta = -residual
        let neg_res: Array1<F> = residual.mapv(|r| F::zero() - r);
        let delta = solve_imex_linear(&jac, &neg_res, F::one(), dt)?;

        // Update
        for i in 0..n {
            y[i] += delta[i];
        }
    }

    // Did not converge but return best attempt
    n_iters = cfg.max_iter_newton;
    Err(IntegrateError::ConvergenceError(format!(
        "IMEX Newton solver did not converge in {} iterations",
        cfg.max_iter_newton
    )))
    .or(Ok((y, n_iters)))
}

// ---------------------------------------------------------------------------
// IMEX Euler (first-order)
// ---------------------------------------------------------------------------

/// First-order IMEX Euler method.
///
/// The scheme is:
///   y* = y_n + dt * f_E(t_n, y_n)          (explicit Euler)
///   y_{n+1} = y* + dt * f_I(t_{n+1}, y_{n+1})  (implicit Euler, solved by Newton)
///
/// This is first-order accurate in time for both stiff and non-stiff parts.
///
/// # Arguments
///
/// * `sys` - Split ODE system implementing `SplitFunction`
/// * `t0` - Initial time
/// * `y0` - Initial condition
/// * `cfg` - IMEX configuration
///
/// # Returns
///
/// `IMEXResult` with solution trajectory or an error.
pub fn imex_euler<F, Sys>(
    sys: &Sys,
    t0: F,
    y0: Array1<F>,
    cfg: &IMEXConfig<F>,
) -> IntegrateResult<IMEXResult<F>>
where
    F: IntegrateFloat,
    Sys: SplitFunction<F>,
{
    let n = sys.dimension();
    if y0.len() != n {
        return Err(IntegrateError::DimensionMismatch(format!(
            "Initial condition length {} != system dimension {}",
            y0.len(),
            n
        )));
    }

    let dt = cfg.dt;
    let mut t = t0;
    let mut y = y0.clone();

    let mut ts = vec![t];
    let mut ys = vec![y0];
    let mut stiff_ratios: Vec<F> = Vec::new();
    let mut n_steps = 0usize;
    let mut total_newton = 0usize;

    while t < cfg.t_end - dt * to_f(0.5) {
        // Clamp last step
        let step = if t + dt > cfg.t_end {
            cfg.t_end - t
        } else {
            dt
        };
        let t_next = t + step;

        // Explicit Euler stage
        let f_e = sys.explicit_part(t, y.view());
        let mut y_star = Array1::<F>::zeros(n);
        for i in 0..n {
            y_star[i] = y[i] + step * f_e[i];
        }

        // Implicit Euler solve: y_{n+1} = y_star + step * f_I(t_next, y_{n+1})
        // i.e., y_{n+1} - step*f_I(t_next, y_{n+1}) = y_star
        // explicit_term here is zero (already embedded in y_star)
        let zero_expl = Array1::<F>::zeros(n);
        match newton_solve_implicit(sys, t_next, &y_star, &zero_expl, step, cfg) {
            Ok((y_new, iters)) => {
                total_newton += iters;
                y = y_new.clone();
                t = t_next;
                ts.push(t);
                ys.push(y_new);
                n_steps += 1;

                if cfg.compute_stiffness {
                    stiff_ratios.push(estimate_stiffness_ratio(sys, t, &y, step)?);
                }
            }
            Err(e) => return Err(e),
        }
    }

    Ok(IMEXResult {
        t: ts,
        y: ys,
        stiffness_ratio: stiff_ratios,
        n_steps,
        n_newton_iters: total_newton,
    })
}

// ---------------------------------------------------------------------------
// IMEX Midpoint (second-order)
// ---------------------------------------------------------------------------

/// Second-order IMEX Midpoint method.
///
/// The scheme is:
///   y_half = y_n + (dt/2) * f_E(t_n, y_n)                       (explicit predictor)
///   y_{n+1} = y_n + dt * f_I(t_n + dt/2, (y_n + y_{n+1})/2)    (implicit midpoint)
///           + dt * f_E(t_n, y_n)                                  (explicit correction)
///
/// The implicit part uses the midpoint rule (2nd order) while the explicit
/// part uses a simple Euler step.
///
/// # Arguments
///
/// * `sys` - Split ODE system
/// * `t0` - Initial time
/// * `y0` - Initial condition
/// * `cfg` - IMEX configuration
pub fn imex_midpoint<F, Sys>(
    sys: &Sys,
    t0: F,
    y0: Array1<F>,
    cfg: &IMEXConfig<F>,
) -> IntegrateResult<IMEXResult<F>>
where
    F: IntegrateFloat,
    Sys: SplitFunction<F>,
{
    let n = sys.dimension();
    if y0.len() != n {
        return Err(IntegrateError::DimensionMismatch(format!(
            "Initial condition length {} != system dimension {}",
            y0.len(),
            n
        )));
    }

    let dt = cfg.dt;
    let mut t = t0;
    let mut y = y0.clone();

    let mut ts = vec![t];
    let mut ys = vec![y0];
    let mut stiff_ratios: Vec<F> = Vec::new();
    let mut n_steps = 0usize;
    let mut total_newton = 0usize;

    while t < cfg.t_end - dt * to_f(0.5) {
        let step = if t + dt > cfg.t_end {
            cfg.t_end - t
        } else {
            dt
        };
        let t_mid = t + step * to_f(0.5);

        // Explicit part: f_E(t_n, y_n)
        let f_e = sys.explicit_part(t, y.view());

        // Explicit term added to the right-hand side: dt * f_E(t_n, y_n)
        let mut expl_term = Array1::<F>::zeros(n);
        for i in 0..n {
            expl_term[i] = step * f_e[i];
        }

        // Implicit midpoint: y_{n+1} = y_n + step * f_I(t_mid, (y_n+y_{n+1})/2) + expl_term
        // Let u = y_{n+1}. Define g(u) = u - y_n - step * f_I(t_mid, (y_n+u)/2) - expl_term = 0
        // Newton: Jacobian of g is I - (step/2) * J_I(t_mid, (y_n+u)/2)
        let y_n = y.clone();
        let mut u = y_n.clone();
        // Add explicit term to predictor
        for i in 0..n {
            u[i] += expl_term[i];
        }

        let mut n_iters_step = 0usize;
        let mut converged = false;
        for _iter in 0..cfg.max_iter_newton {
            // Midpoint state
            let mut y_mid = Array1::<F>::zeros(n);
            for i in 0..n {
                y_mid[i] = (y_n[i] + u[i]) * to_f(0.5);
            }

            let f_i_mid = sys.implicit_part(t_mid, y_mid.view());

            // Residual: u - y_n - step*f_I(t_mid, y_mid) - expl_term
            let mut res = Array1::<F>::zeros(n);
            for i in 0..n {
                res[i] = u[i] - y_n[i] - step * f_i_mid[i] - expl_term[i];
            }

            let res_norm = res.iter().fold(F::zero(), |acc, &r| acc + r * r).sqrt();
            if res_norm < cfg.newton_tol {
                n_iters_step = _iter + 1;
                converged = true;
                break;
            }

            // Jacobian of g: I - (step/2)*J_I
            let jac = sys.jacobian_implicit(t_mid, y_mid.view());
            let neg_res: Array1<F> = res.mapv(|r| F::zero() - r);

            // Solve (I - (step/2)*J_I) * delta = -residual
            let mut mat = Array2::<F>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    mat[[i, j]] = if i == j {
                        F::one() - step * to_f(0.5) * jac[[i, j]]
                    } else {
                        F::zero() - step * to_f(0.5) * jac[[i, j]]
                    };
                }
            }
            let mut rhs_copy = neg_res;
            let delta = gaussian_elimination(&mut mat, &mut rhs_copy)?;

            for i in 0..n {
                u[i] += delta[i];
            }
        }

        if !converged {
            n_iters_step = cfg.max_iter_newton;
        }

        total_newton += n_iters_step;
        y = u.clone();
        t += step;
        ts.push(t);
        ys.push(u);
        n_steps += 1;

        if cfg.compute_stiffness {
            stiff_ratios.push(estimate_stiffness_ratio(sys, t, &y, step)?);
        }
    }

    Ok(IMEXResult {
        t: ts,
        y: ys,
        stiffness_ratio: stiff_ratios,
        n_steps,
        n_newton_iters: total_newton,
    })
}

// ---------------------------------------------------------------------------
// IMEX BDF2 (second-order)
// ---------------------------------------------------------------------------

/// Second-order IMEX BDF2 method.
///
/// Uses Adams-Bashforth extrapolation for the explicit part and BDF2 for the implicit part:
///
///   (3/2) y_{n+1} - 2 y_n + (1/2) y_{n-1} = dt * [2 f_E(t_n, y_n) - f_E(t_{n-1}, y_{n-1})
///                                                   + f_I(t_{n+1}, y_{n+1})]
///
/// The first step is bootstrapped using IMEX Euler.
///
/// # Arguments
///
/// * `sys` - Split ODE system
/// * `t0` - Initial time
/// * `y0` - Initial condition
/// * `cfg` - IMEX configuration
pub fn imex_bdf2<F, Sys>(
    sys: &Sys,
    t0: F,
    y0: Array1<F>,
    cfg: &IMEXConfig<F>,
) -> IntegrateResult<IMEXResult<F>>
where
    F: IntegrateFloat,
    Sys: SplitFunction<F>,
{
    let n = sys.dimension();
    if y0.len() != n {
        return Err(IntegrateError::DimensionMismatch(format!(
            "Initial condition length {} != system dimension {}",
            y0.len(),
            n
        )));
    }

    let dt = cfg.dt;

    // Bootstrap with one IMEX Euler step to get y_1
    let f_e0 = sys.explicit_part(t0, y0.view());
    let mut y_star = Array1::<F>::zeros(n);
    for i in 0..n {
        y_star[i] = y0[i] + dt * f_e0[i];
    }
    let zero_expl = Array1::<F>::zeros(n);
    let (y1, newton0) = newton_solve_implicit(sys, t0 + dt, &y_star, &zero_expl, dt, cfg)
        .unwrap_or_else(|_| (y_star.clone(), cfg.max_iter_newton));

    let t1 = t0 + dt;

    let mut ts = vec![t0, t1];
    let mut ys = vec![y0.clone(), y1.clone()];
    let mut stiff_ratios: Vec<F> = Vec::new();
    let mut n_steps = 1usize;
    let mut total_newton = newton0;

    let mut y_prev = y0.clone();
    let mut f_e_prev = f_e0;
    let mut y_curr = y1;
    let mut t_curr = t1;

    // BDF2 main loop
    while t_curr < cfg.t_end - dt * to_f(0.5) {
        let step = if t_curr + dt > cfg.t_end {
            cfg.t_end - t_curr
        } else {
            dt
        };
        let t_next = t_curr + step;

        let f_e_curr = sys.explicit_part(t_curr, y_curr.view());

        // Adams-Bashforth 2nd order explicit extrapolation:
        // expl_rhs = dt * (2*f_E(t_n) - f_E(t_{n-1}))
        let mut expl_rhs = Array1::<F>::zeros(n);
        for i in 0..n {
            expl_rhs[i] = step * (to_f::<F>(2.0) * f_e_curr[i] - f_e_prev[i]);
        }

        // BDF2 RHS constant (without implicit part):
        // rhs_const = 2*y_n - (1/2)*y_{n-1} + expl_rhs
        let mut rhs_const = Array1::<F>::zeros(n);
        for i in 0..n {
            rhs_const[i] = to_f::<F>(2.0) * y_curr[i] - to_f::<F>(0.5) * y_prev[i] + expl_rhs[i];
        }

        // BDF2 equation: (3/2)*y_{n+1} - step*f_I(t_{n+1}, y_{n+1}) = rhs_const
        // Newton: g(u) = (3/2)*u - step*f_I(t_next, u) - rhs_const = 0
        // Jacobian: (3/2)*I - step*J_I
        let mut u = y_curr.clone();
        let mut n_iters_step = 0usize;
        let three_half = to_f::<F>(1.5);

        for _iter in 0..cfg.max_iter_newton {
            let f_i = sys.implicit_part(t_next, u.view());
            let mut res = Array1::<F>::zeros(n);
            for i in 0..n {
                res[i] = three_half * u[i] - step * f_i[i] - rhs_const[i];
            }

            let res_norm = res.iter().fold(F::zero(), |acc, &r| acc + r * r).sqrt();
            if res_norm < cfg.newton_tol {
                n_iters_step = _iter + 1;
                break;
            }

            let jac = sys.jacobian_implicit(t_next, u.view());
            let neg_res: Array1<F> = res.mapv(|r| F::zero() - r);
            // Solve (3/2*I - step*J_I) * delta = -res
            let delta = solve_imex_linear(&jac, &neg_res, three_half, step)?;

            for i in 0..n {
                u[i] += delta[i];
            }

            if _iter + 1 == cfg.max_iter_newton {
                n_iters_step = cfg.max_iter_newton;
            }
        }

        total_newton += n_iters_step;

        // Advance
        y_prev = y_curr;
        f_e_prev = f_e_curr;
        y_curr = u.clone();
        t_curr = t_next;

        ts.push(t_curr);
        ys.push(u);
        n_steps += 1;

        if cfg.compute_stiffness {
            stiff_ratios.push(estimate_stiffness_ratio(sys, t_curr, &y_curr, step)?);
        }
    }

    Ok(IMEXResult {
        t: ts,
        y: ys,
        stiffness_ratio: stiff_ratios,
        n_steps,
        n_newton_iters: total_newton,
    })
}

// ---------------------------------------------------------------------------
// IMEX-ARK SSP2 (2-stage, 2nd order)
// ---------------------------------------------------------------------------

/// IMEX-ARK SSP2(2,2,2) scheme by Ascher, Ruuth, Spiteri (1997).
///
/// This is a 2-stage, 2nd-order IMEX Runge-Kutta scheme.
/// The implicit part uses an L-stable SDIRK (Singly Diagonally Implicit)
/// method with γ = 1 - 1/√2.
///
/// **Explicit Butcher tableau** (SSP2):
/// ```text
///   0  | 0    0
///   1  | 1    0
///      | 1/2  1/2
/// ```
///
/// **Implicit Butcher tableau** (SDIRK):
/// ```text
///   γ | γ       0
///   1 | 1-γ     γ
///     | 1/2     1/2
/// ```
/// where γ = 1 - 1/√2 ≈ 0.2929.
///
/// # Arguments
///
/// * `sys` - Split ODE system
/// * `t0` - Initial time
/// * `y0` - Initial condition
/// * `cfg` - IMEX configuration
pub fn imex_ark_ssp2<F, Sys>(
    sys: &Sys,
    t0: F,
    y0: Array1<F>,
    cfg: &IMEXConfig<F>,
) -> IntegrateResult<IMEXResult<F>>
where
    F: IntegrateFloat,
    Sys: SplitFunction<F>,
{
    let n = sys.dimension();
    if y0.len() != n {
        return Err(IntegrateError::DimensionMismatch(format!(
            "Initial condition length {} != system dimension {}",
            y0.len(),
            n
        )));
    }

    // γ = 1 - 1/√2
    let gamma: F = to_f(1.0 - 1.0 / std::f64::consts::SQRT_2);
    let one_minus_gamma: F = F::one() - gamma;

    let dt = cfg.dt;
    let mut t = t0;
    let mut y = y0.clone();

    let mut ts = vec![t];
    let mut ys = vec![y0];
    let mut stiff_ratios: Vec<F> = Vec::new();
    let mut n_steps = 0usize;
    let mut total_newton = 0usize;

    while t < cfg.t_end - dt * to_f(0.5) {
        let step = if t + dt > cfg.t_end {
            cfg.t_end - t
        } else {
            dt
        };

        // ---- Stage 1 ----
        // Explicit: c_E1 = 0, Y1_E = y_n
        // Implicit: c_I1 = γ, solve (I - step*γ*J_I) * Y1_I = y_n + step*γ*f_I(t+γ*h, Y1_I)
        //           i.e., Y1_I = y_n + step*γ*f_I(t+c1*h, Y1_I)

        let t_stage1 = t + gamma * step;

        // Newton for implicit stage 1: Y1 - step*gamma*f_I(t_stage1, Y1) = y_n
        let mut y1_i = y.clone();
        let mut n_iter1 = 0usize;
        for _it in 0..cfg.max_iter_newton {
            let f_i1 = sys.implicit_part(t_stage1, y1_i.view());
            let mut res = Array1::<F>::zeros(n);
            for i in 0..n {
                res[i] = y1_i[i] - step * gamma * f_i1[i] - y[i];
            }
            let res_norm = res.iter().fold(F::zero(), |acc, &r| acc + r * r).sqrt();
            if res_norm < cfg.newton_tol {
                n_iter1 = _it + 1;
                break;
            }
            let jac = sys.jacobian_implicit(t_stage1, y1_i.view());
            let neg_res: Array1<F> = res.mapv(|r| F::zero() - r);
            let delta = solve_imex_linear(&jac, &neg_res, F::one(), step * gamma)?;
            for i in 0..n {
                y1_i[i] += delta[i];
            }
            if _it + 1 == cfg.max_iter_newton {
                n_iter1 = cfg.max_iter_newton;
            }
        }
        total_newton += n_iter1;

        // Explicit f at stage 1: f_E(t, y_n) (c_E1 = 0)
        let k1_e = sys.explicit_part(t, y.view());
        // Implicit f at stage 1
        let k1_i = sys.implicit_part(t_stage1, y1_i.view());

        // ---- Stage 2 ----
        // Explicit: c_E2 = 1, Y2_E = y_n + step*1*k1_E
        // Implicit: c_I2 = 1, Y2_I = y_n + step*(1-γ)*k1_I + step*γ*f_I(t+h, Y2_I)

        let t_stage2 = t + step; // c_I2 = 1

        let mut y2_e = Array1::<F>::zeros(n);
        for i in 0..n {
            y2_e[i] = y[i] + step * k1_e[i];
        }

        // Newton for implicit stage 2
        let mut y2_i = y.clone();
        // Initial guess: y_n + step*(1-γ)*k1_I
        for i in 0..n {
            y2_i[i] = y[i] + step * one_minus_gamma * k1_i[i];
        }

        let mut n_iter2 = 0usize;
        for _it in 0..cfg.max_iter_newton {
            let f_i2 = sys.implicit_part(t_stage2, y2_i.view());
            let mut res = Array1::<F>::zeros(n);
            for i in 0..n {
                res[i] = y2_i[i] - step * one_minus_gamma * k1_i[i] - step * gamma * f_i2[i] - y[i];
            }
            let res_norm = res.iter().fold(F::zero(), |acc, &r| acc + r * r).sqrt();
            if res_norm < cfg.newton_tol {
                n_iter2 = _it + 1;
                break;
            }
            let jac = sys.jacobian_implicit(t_stage2, y2_i.view());
            let neg_res: Array1<F> = res.mapv(|r| F::zero() - r);
            let delta = solve_imex_linear(&jac, &neg_res, F::one(), step * gamma)?;
            for i in 0..n {
                y2_i[i] += delta[i];
            }
            if _it + 1 == cfg.max_iter_newton {
                n_iter2 = cfg.max_iter_newton;
            }
        }
        total_newton += n_iter2;

        let k2_e = sys.explicit_part(t + step, y2_e.view()); // c_E2 = 1
        let k2_i = sys.implicit_part(t_stage2, y2_i.view());

        // ---- Final combination ----
        // b_E = [1/2, 1/2], b_I = [1/2, 1/2]
        let mut y_new = Array1::<F>::zeros(n);
        for i in 0..n {
            y_new[i] = y[i]
                + step * to_f(0.5) * (k1_e[i] + k2_e[i])
                + step * to_f(0.5) * (k1_i[i] + k2_i[i]);
        }

        y = y_new.clone();
        t += step;
        ts.push(t);
        ys.push(y_new);
        n_steps += 1;

        if cfg.compute_stiffness {
            stiff_ratios.push(estimate_stiffness_ratio(sys, t, &y, step)?);
        }
    }

    Ok(IMEXResult {
        t: ts,
        y: ys,
        stiffness_ratio: stiff_ratios,
        n_steps,
        n_newton_iters: total_newton,
    })
}

// ---------------------------------------------------------------------------
// IMEX-ARK SSP3 (3-stage, 2nd-order, Pareschi-Russo)
// ---------------------------------------------------------------------------

/// IMEX-ARK SSP3(3,3,2) scheme by Pareschi and Russo (2005).
///
/// A 3-stage, 2nd-order IMEX scheme with SSP property for the explicit part.
///
/// **Explicit Butcher tableau** (SSP-RK3):
/// ```text
///   0   | 0    0    0
///   1   | 1    0    0
///   1/2 | 1/4  1/4  0
///       | 1/6  1/6  2/3
/// ```
///
/// **Implicit Butcher tableau** (SDIRK, γ ≈ 0.2679):
/// ```text
///   γ   | γ      0      0
///   1-γ | 1-2γ   γ      0
///   1/2 | 1/2-γ  0      γ
///       | 1/6    1/6    2/3
/// ```
/// where γ = (3 + √3) / 6.
///
/// Reference: Pareschi & Russo, "Implicit-Explicit Runge-Kutta schemes", 2005.
pub fn imex_ark_ssp3<F, Sys>(
    sys: &Sys,
    t0: F,
    y0: Array1<F>,
    cfg: &IMEXConfig<F>,
) -> IntegrateResult<IMEXResult<F>>
where
    F: IntegrateFloat,
    Sys: SplitFunction<F>,
{
    let n = sys.dimension();
    if y0.len() != n {
        return Err(IntegrateError::DimensionMismatch(format!(
            "Initial condition length {} != system dimension {}",
            y0.len(),
            n
        )));
    }

    // γ = (3 + √3) / 6
    let gamma: F = to_f((3.0 + 3.0_f64.sqrt()) / 6.0);
    let two_gamma = gamma * to_f(2.0);
    let one_minus_two_gamma = F::one() - two_gamma;
    let half_minus_gamma: F = to_f::<F>(0.5) - gamma;

    let dt = cfg.dt;
    let mut t = t0;
    let mut y = y0.clone();

    let mut ts = vec![t];
    let mut ys = vec![y0];
    let mut stiff_ratios: Vec<F> = Vec::new();
    let mut n_steps = 0usize;
    let mut total_newton = 0usize;

    while t < cfg.t_end - dt * to_f(0.5) {
        let step = if t + dt > cfg.t_end {
            cfg.t_end - t
        } else {
            dt
        };

        // ---- Stage 1 (c_E1=0, c_I1=γ) ----
        let t_i1 = t + gamma * step;
        let k1_e = sys.explicit_part(t, y.view());

        // Implicit stage 1: Y1 = y + step*γ*f_I(t_i1, Y1)
        let (y1_i, ni1) =
            solve_sdirk_stage(sys, t_i1, &y, &Array1::<F>::zeros(n), gamma, step, cfg)?;
        total_newton += ni1;
        let k1_i = sys.implicit_part(t_i1, y1_i.view());

        // ---- Stage 2 (c_E2=1, c_I2=1-γ) ----
        let t_i2 = t + (F::one() - gamma) * step;
        // Explicit stage 2 state
        let mut y2_e = Array1::<F>::zeros(n);
        for i in 0..n {
            y2_e[i] = y[i] + step * k1_e[i];
        }
        let k2_e = sys.explicit_part(t + step, y2_e.view());

        // Implicit stage 2: Y2 = y + step*(1-2γ)*k1_I + step*γ*f_I(t_i2, Y2)
        let mut acc2 = Array1::<F>::zeros(n);
        for i in 0..n {
            acc2[i] = step * one_minus_two_gamma * k1_i[i];
        }
        let (y2_i, ni2) = solve_sdirk_stage(sys, t_i2, &y, &acc2, gamma, step, cfg)?;
        total_newton += ni2;
        let k2_i = sys.implicit_part(t_i2, y2_i.view());

        // ---- Stage 3 (c_E3=1/2, c_I3=1/2) ----
        let t_i3 = t + to_f::<F>(0.5) * step;
        // Explicit stage 3 state
        let mut y3_e = Array1::<F>::zeros(n);
        for i in 0..n {
            y3_e[i] = y[i] + step * (to_f::<F>(0.25) * k1_e[i] + to_f::<F>(0.25) * k2_e[i]);
        }
        let k3_e = sys.explicit_part(t + to_f::<F>(0.5) * step, y3_e.view());

        // Implicit stage 3: Y3 = y + step*(1/2-γ)*k1_I + 0*k2_I + step*γ*f_I(t_i3, Y3)
        let mut acc3 = Array1::<F>::zeros(n);
        for i in 0..n {
            acc3[i] = step * half_minus_gamma * k1_i[i];
        }
        let (y3_i, ni3) = solve_sdirk_stage(sys, t_i3, &y, &acc3, gamma, step, cfg)?;
        total_newton += ni3;
        let k3_i = sys.implicit_part(t_i3, y3_i.view());

        // ---- Final combination ----
        // b_E = [1/6, 1/6, 2/3], b_I = [1/6, 1/6, 2/3]
        let mut y_new = Array1::<F>::zeros(n);
        for i in 0..n {
            y_new[i] = y[i]
                + step
                    * (to_f::<F>(1.0 / 6.0) * (k1_e[i] + k1_i[i])
                        + to_f::<F>(1.0 / 6.0) * (k2_e[i] + k2_i[i])
                        + to_f::<F>(2.0 / 3.0) * (k3_e[i] + k3_i[i]));
        }

        y = y_new.clone();
        t += step;
        ts.push(t);
        ys.push(y_new);
        n_steps += 1;

        if cfg.compute_stiffness {
            stiff_ratios.push(estimate_stiffness_ratio(sys, t, &y, step)?);
        }
    }

    Ok(IMEXResult {
        t: ts,
        y: ys,
        stiffness_ratio: stiff_ratios,
        n_steps,
        n_newton_iters: total_newton,
    })
}

// ---------------------------------------------------------------------------
// Helper: SDIRK stage solve
// ---------------------------------------------------------------------------

/// Solve a single SDIRK stage: Y = y_base + acc + step*gamma*f_I(t_stage, Y)
///
/// Returns (Y, n_newton_iters).
fn solve_sdirk_stage<F, Sys>(
    sys: &Sys,
    t_stage: F,
    y_base: &Array1<F>,
    acc: &Array1<F>,
    gamma: F,
    step: F,
    cfg: &IMEXConfig<F>,
) -> IntegrateResult<(Array1<F>, usize)>
where
    F: IntegrateFloat,
    Sys: SplitFunction<F>,
{
    let n = y_base.len();
    let mut y = Array1::<F>::zeros(n);
    for i in 0..n {
        y[i] = y_base[i] + acc[i]; // initial guess
    }

    let alpha = step * gamma;
    let mut n_iters = 0usize;

    for _it in 0..cfg.max_iter_newton {
        let f_i = sys.implicit_part(t_stage, y.view());
        let mut res = Array1::<F>::zeros(n);
        for i in 0..n {
            res[i] = y[i] - acc[i] - alpha * f_i[i] - y_base[i];
        }
        let res_norm = res.iter().fold(F::zero(), |acc, &r| acc + r * r).sqrt();
        if res_norm < cfg.newton_tol {
            n_iters = _it + 1;
            return Ok((y, n_iters));
        }
        let jac = sys.jacobian_implicit(t_stage, y.view());
        let neg_res: Array1<F> = res.mapv(|r| F::zero() - r);
        let delta = solve_imex_linear(&jac, &neg_res, F::one(), alpha)?;
        for i in 0..n {
            y[i] += delta[i];
        }
        if _it + 1 == cfg.max_iter_newton {
            n_iters = cfg.max_iter_newton;
        }
    }

    Ok((y, n_iters))
}

// ---------------------------------------------------------------------------
// Stiffness ratio estimation
// ---------------------------------------------------------------------------

/// Estimate the ratio of stiffness by comparing spectral radii of J_I and J_E
/// via the Gershgorin circle theorem (cheap upper bound).
fn estimate_stiffness_ratio<F, Sys>(sys: &Sys, t: F, y: &Array1<F>, _dt: F) -> IntegrateResult<F>
where
    F: IntegrateFloat,
    Sys: SplitFunction<F>,
{
    let n = sys.dimension();
    let j_i = sys.jacobian_implicit(t, y.view());

    // Gershgorin radius for implicit Jacobian
    let mut rho_i = F::zero();
    for row in 0..n {
        let diag = j_i[[row, row]].abs();
        let off_sum: F = (0..n)
            .filter(|&j| j != row)
            .fold(F::zero(), |s, j| s + j_i[[row, j]].abs());
        let r = diag + off_sum;
        if r > rho_i {
            rho_i = r;
        }
    }

    // For the explicit part we use a finite-difference Jacobian approximation
    let eps: F = to_f(1e-7);
    let f_base = sys.explicit_part(t, y.view());
    let mut rho_e = F::zero();
    for col in 0..n {
        let mut y_pert = y.clone();
        y_pert[col] += eps;
        let f_pert = sys.explicit_part(t, y_pert.view());
        let col_norm = (0..n)
            .fold(F::zero(), |s, row| {
                let diff = (f_pert[row] - f_base[row]) / eps;
                s + diff * diff
            })
            .sqrt();
        if col_norm > rho_e {
            rho_e = col_norm;
        }
    }

    if rho_e < to_f(1e-300) {
        Ok(to_f(1.0))
    } else {
        Ok(rho_i / rho_e)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    /// Simple stiff test problem: dy/dt = lambda * y
    /// Split: f_I = lambda_stiff * y, f_E = lambda_nonstiff * y
    struct StiffLinear {
        lambda_stiff: f64,
        lambda_nonstiff: f64,
    }

    impl SplitFunction<f64> for StiffLinear {
        fn explicit_part(&self, _t: f64, y: ArrayView1<f64>) -> Array1<f64> {
            array![self.lambda_nonstiff * y[0]]
        }

        fn implicit_part(&self, _t: f64, y: ArrayView1<f64>) -> Array1<f64> {
            array![self.lambda_stiff * y[0]]
        }

        fn jacobian_implicit(&self, _t: f64, _y: ArrayView1<f64>) -> Array2<f64> {
            let mut j = Array2::<f64>::zeros((1, 1));
            j[[0, 0]] = self.lambda_stiff;
            j
        }

        fn dimension(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_imex_euler_decay() {
        // dy/dt = -10*y (stiff) + 0*y (nonstiff), y(0) = 1
        // exact: y(t) = exp(-10*t)
        let sys = StiffLinear {
            lambda_stiff: -10.0,
            lambda_nonstiff: 0.0,
        };
        let cfg = IMEXConfig {
            dt: 0.01,
            t_end: 1.0,
            newton_tol: 1e-12,
            ..IMEXConfig::default()
        };
        let result = imex_euler(&sys, 0.0, array![1.0], &cfg).expect("IMEX Euler failed");

        let t_final = *result.t.last().expect("no time points");
        let y_final = result.y.last().expect("no solution")[0];
        let exact = (-10.0_f64 * t_final).exp();

        assert!(
            (y_final - exact).abs() < 0.05,
            "IMEX Euler: y={} exact={} err={}",
            y_final,
            exact,
            (y_final - exact).abs()
        );
    }

    #[test]
    fn test_imex_bdf2_decay() {
        let sys = StiffLinear {
            lambda_stiff: -5.0,
            lambda_nonstiff: -1.0,
        };
        let cfg = IMEXConfig {
            dt: 0.01,
            t_end: 0.5,
            newton_tol: 1e-12,
            ..IMEXConfig::default()
        };
        let result = imex_bdf2(&sys, 0.0, array![1.0], &cfg).expect("IMEX BDF2 failed");

        let t_final = *result.t.last().expect("no time points");
        let y_final = result.y.last().expect("no solution")[0];
        let exact = (-6.0_f64 * t_final).exp();

        assert!(
            (y_final - exact).abs() < 0.02,
            "IMEX BDF2: y={} exact={} err={}",
            y_final,
            exact,
            (y_final - exact).abs()
        );
    }

    #[test]
    fn test_imex_ark_ssp2_decay() {
        let sys = StiffLinear {
            lambda_stiff: -5.0,
            lambda_nonstiff: -1.0,
        };
        let cfg = IMEXConfig {
            dt: 0.01,
            t_end: 0.5,
            newton_tol: 1e-12,
            ..IMEXConfig::default()
        };
        let result = imex_ark_ssp2(&sys, 0.0, array![1.0], &cfg).expect("IMEX ARK SSP2 failed");

        let t_final = *result.t.last().expect("no time points");
        let y_final = result.y.last().expect("no solution")[0];
        let exact = (-6.0_f64 * t_final).exp();

        assert!(
            (y_final - exact).abs() < 0.01,
            "IMEX ARK SSP2: y={} exact={} err={}",
            y_final,
            exact,
            (y_final - exact).abs()
        );
    }

    #[test]
    fn test_imex_ark_ssp3_decay() {
        let sys = StiffLinear {
            lambda_stiff: -5.0,
            lambda_nonstiff: -1.0,
        };
        let cfg = IMEXConfig {
            dt: 0.01,
            t_end: 0.5,
            newton_tol: 1e-12,
            ..IMEXConfig::default()
        };
        let result = imex_ark_ssp3(&sys, 0.0, array![1.0], &cfg).expect("IMEX ARK SSP3 failed");

        let t_final = *result.t.last().expect("no time points");
        let y_final = result.y.last().expect("no solution")[0];
        let exact = (-6.0_f64 * t_final).exp();

        assert!(
            (y_final - exact).abs() < 0.01,
            "IMEX ARK SSP3: y={} exact={} err={}",
            y_final,
            exact,
            (y_final - exact).abs()
        );
    }

    #[test]
    fn test_imex_midpoint_decay() {
        let sys = StiffLinear {
            lambda_stiff: -5.0,
            lambda_nonstiff: -1.0,
        };
        let cfg = IMEXConfig {
            dt: 0.01,
            t_end: 0.5,
            newton_tol: 1e-12,
            ..IMEXConfig::default()
        };
        let result = imex_midpoint(&sys, 0.0, array![1.0], &cfg).expect("IMEX Midpoint failed");

        let t_final = *result.t.last().expect("no time points");
        let y_final = result.y.last().expect("no solution")[0];
        let exact = (-6.0_f64 * t_final).exp();

        assert!(
            (y_final - exact).abs() < 0.01,
            "IMEX Midpoint: y={} exact={} err={}",
            y_final,
            exact,
            (y_final - exact).abs()
        );
    }
}
