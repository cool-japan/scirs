//! Explicit ODE solver methods
//!
//! This module implements explicit methods for solving ODEs,
//! including Euler's method and the classic 4th-order Runge-Kutta method.

use crate::error::IntegrateResult;
use crate::ode::types::{ODEMethod, ODEOptions, ODEResult};
use crate::IntegrateFloat;
use scirs2_core::ndarray::{Array1, ArrayView1};

/// Solve ODE using Euler's method
///
/// This is the simplest numerical method for solving ODEs, with first-order accuracy.
/// It is included primarily for educational purposes and is not recommended for
/// practical use due to its low accuracy and efficiency.
///
/// # Arguments
///
/// * `f` - ODE function dy/dt = f(t, y)
/// * `t_span` - Time span [t_start, t_end]
/// * `y0` - Initial condition
/// * `h` - Step size
/// * `opts` - Solver options
///
/// # Returns
///
/// The solution as an ODEResult or an error
#[allow(dead_code)]
pub fn euler_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    h: F,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Initialize
    let [t_start, t_end] = t_span;
    let step_size = h;

    // Compute number of steps
    let mut t = t_start;
    let mut y = y0.clone();

    // Storage for results
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];

    // Statistics
    let mut func_evals = 0;
    let mut step_count = 0;

    // Main integration loop
    while t < t_end && step_count < opts.max_steps {
        // Calculate next time point
        let next_t = if t + step_size > t_end {
            t_end
        } else {
            t + step_size
        };

        // Calculate step size for this iteration
        let h_actual = next_t - t;

        // Compute the derivative at the current point
        let dy = f(t, y.view());
        func_evals += 1;

        // Euler step: y_{n+1} = y_n + h * f(t_n, y_n)
        let y_next = y.clone() + dy * h_actual;

        // Store results
        t = next_t;
        y = y_next;
        t_values.push(t);
        y_values.push(y.clone());

        step_count += 1;
    }

    // Check if integration was successful
    let success = t >= t_end;
    let message = if !success {
        Some(format!(
            "Maximum number of steps ({}) reached",
            opts.max_steps
        ))
    } else {
        None
    };

    // Return the solution
    Ok(ODEResult {
        t: t_values,
        y: y_values,
        success,
        message,
        n_eval: func_evals,
        n_steps: step_count,
        n_accepted: step_count, // All steps are accepted in fixed-step methods
        n_rejected: 0,          // No steps are rejected in fixed-step methods
        n_lu: 0,                // No LU decompositions in explicit methods
        n_jac: 0,               // No Jacobian evaluations in explicit methods
        method: ODEMethod::Euler,
    })
}

/// Solve ODE using the classical 4th-order Runge-Kutta method
///
/// This is a popular fixed-step size method with 4th-order accuracy.
/// It provides a good balance between simplicity and accuracy for non-stiff problems.
///
/// # Arguments
///
/// * `f` - ODE function dy/dt = f(t, y)
/// * `t_span` - Time span [t_start, t_end]
/// * `y0` - Initial condition
/// * `h` - Step size
/// * `opts` - Solver options
///
/// # Returns
///
/// The solution as an ODEResult or an error
#[allow(dead_code)]
pub fn rk4_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    h: F,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Initialize
    let [t_start, t_end] = t_span;
    let step_size = h;

    // Compute number of steps
    let mut t = t_start;
    let mut y = y0.clone();

    // Storage for results
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];

    // Statistics
    let mut func_evals = 0;
    let mut step_count = 0;

    // Constants for RK4
    let two = F::from_f64(2.0).expect("Operation failed");
    let six = F::from_f64(6.0).expect("Operation failed");

    // Main integration loop
    while t < t_end && step_count < opts.max_steps {
        // Calculate next time point
        let next_t = if t + step_size > t_end {
            t_end
        } else {
            t + step_size
        };

        // Calculate step size for this iteration
        let h_actual = next_t - t;
        let half_step = h_actual / two;

        // RK4 stages
        let k1 = f(t, y.view());
        let k2 = f(t + half_step, (y.clone() + k1.clone() * half_step).view());
        let k3 = f(t + half_step, (y.clone() + k2.clone() * half_step).view());
        let k4 = f(t + h_actual, (y.clone() + k3.clone() * h_actual).view());
        func_evals += 4;

        // Combine the stages with appropriate weights
        let slope = (k1 + k2.clone() * two + k3.clone() * two + k4) / six;

        // RK4 step: y_{n+1} = y_n + h * (k1 + 2*k2 + 2*k3 + k4)/6
        let y_next = y.clone() + slope * h_actual;

        // Store results
        t = next_t;
        y = y_next;
        t_values.push(t);
        y_values.push(y.clone());

        step_count += 1;
    }

    // Check if integration was successful
    let success = t >= t_end;
    let message = if !success {
        Some(format!(
            "Maximum number of steps ({}) reached",
            opts.max_steps
        ))
    } else {
        None
    };

    // Return the solution
    Ok(ODEResult {
        t: t_values,
        y: y_values,
        success,
        message,
        n_eval: func_evals,
        n_steps: step_count,
        n_accepted: step_count, // All steps are accepted in fixed-step methods
        n_rejected: 0,          // No steps are rejected in fixed-step methods
        n_lu: 0,                // No LU decompositions in explicit methods
        n_jac: 0,               // No Jacobian evaluations in explicit methods
        method: ODEMethod::RK4,
    })
}

/// Advance state by one time step using Strong Stability Preserving Runge-Kutta (3rd order).
///
/// Implements the Shu-Osher SSPRK3 scheme (1988):
///
/// ```text
///   u¹   = uⁿ + dt · L(uⁿ, t)
///   u²   = (3/4)·uⁿ + (1/4)·u¹ + (1/4)·dt · L(u¹, t + dt)
///   uⁿ⁺¹ = (1/3)·uⁿ + (2/3)·u² + (2/3)·dt · L(u², t + dt/2)
/// ```
///
/// Strong-stability preservation guarantees the TVD property when the spatial
/// operator `rhs` satisfies it under forward Euler with step `dt`.
///
/// # Arguments
///
/// * `state` - Current state vector (owned or borrowed; cloned internally)
/// * `t`     - Current time
/// * `dt`    - Time step size
/// * `rhs`   - Spatial operator: `L(u, t)` returning the same type as `state`
///
/// # Returns
///
/// New state after one SSPRK3 time step.
///
/// # Example
///
/// ```rust
/// use scirs2_integrate::ode::methods::ssprk3_step;
/// use scirs2_core::ndarray::Array1;
///
/// // Scalar ODE: du/dt = -u  (exponential decay)
/// let state = Array1::from_vec(vec![1.0_f64]);
/// let rhs = |u: &Array1<f64>, _t: f64| -u.clone();
/// let next = ssprk3_step(&state, 0.0, 0.1, &rhs);
/// // next ≈ e^{-0.1} ≈ 0.9048
/// assert!((next[0] - (-0.1_f64).exp()).abs() < 1e-4);
/// ```
pub fn ssprk3_step<S, F>(state: &S, t: f64, dt: f64, rhs: &F) -> S
where
    S: Clone + std::ops::Add<Output = S> + std::ops::Mul<f64, Output = S>,
    F: Fn(&S, f64) -> S,
{
    // Stage 1: u¹ = uⁿ + dt · L(uⁿ, t)
    let l0 = rhs(state, t);
    let u1 = state.clone() + l0 * dt;

    // Stage 2: u² = (3/4)·uⁿ + (1/4)·(u¹ + dt · L(u¹, t + dt))
    let l1 = rhs(&u1, t + dt);
    let u2 = state.clone() * (3.0 / 4.0) + (u1.clone() + l1 * dt) * (1.0 / 4.0);

    // Stage 3: uⁿ⁺¹ = (1/3)·uⁿ + (2/3)·(u² + dt · L(u², t + dt/2))
    let l2 = rhs(&u2, t + 0.5 * dt);
    state.clone() * (1.0 / 3.0) + (u2 + l2 * dt) * (2.0 / 3.0)
}

/// Advance state by one time step using Strong Stability Preserving Runge-Kutta (4th order).
///
/// Implements the optimal SSP(5,4) scheme from Spiteri & Ruuth (2002),
/// "A new class of optimal high-order strong-stability-preserving time
/// discretization methods", SIAM J. Numer. Anal. 40(2):469-491.
///
/// The scheme uses 5 stages to achieve 4th-order accuracy with an SSP
/// coefficient of C = 1.508 (optimal among 5-stage, 4th-order methods).
///
/// # Arguments
///
/// * `state` - Current state vector
/// * `t`     - Current time
/// * `dt`    - Time step size
/// * `rhs`   - Spatial operator: `L(u, t)` returning the same type as `state`
///
/// # Returns
///
/// New state after one SSP(5,4) time step.
///
/// # Example
///
/// ```rust
/// use scirs2_integrate::ode::methods::ssprk4_step;
/// use scirs2_core::ndarray::Array1;
///
/// // Scalar ODE: du/dt = -u  (exponential decay)
/// let state = Array1::from_vec(vec![1.0_f64]);
/// let rhs = |u: &Array1<f64>, _t: f64| -u.clone();
/// let next = ssprk4_step(&state, 0.0, 0.1, &rhs);
/// // 4th-order accurate: next ≈ e^{-0.1} ≈ 0.9048
/// assert!((next[0] - (-0.1_f64).exp()).abs() < 1e-5);
/// ```
pub fn ssprk4_step<S, F>(state: &S, t: f64, dt: f64, rhs: &F) -> S
where
    S: Clone + std::ops::Add<Output = S> + std::ops::Mul<f64, Output = S>,
    F: Fn(&S, f64) -> S,
{
    // Spiteri-Ruuth (2002) optimal SSP(5,4) coefficients.
    // Abscissae (Runge-Kutta c-values):
    const C1: f64 = 0.391_752_226_571_890;
    const C2: f64 = 0.586_079_689_311_540;
    const C3: f64 = 0.474_542_363_121_400;
    const C4: f64 = 0.935_010_630_967_653;

    // Stage 1
    let l0 = rhs(state, t);
    let u1 = state.clone() + l0 * (0.391_752_226_571_890 * dt);

    // Stage 2
    let l1 = rhs(&u1, t + C1 * dt);
    let u2 = state.clone() * 0.444_370_493_651_235
        + u1 * 0.555_629_506_348_765
        + l1 * (0.368_410_593_050_371 * dt);

    // Stage 3
    let l2 = rhs(&u2, t + C2 * dt);
    let u3 = state.clone() * 0.620_101_851_488_403
        + u2.clone() * 0.379_898_148_511_597
        + l2 * (0.251_891_774_271_694 * dt);

    // Stage 4 (reuses L(u3) computed below)
    let l3 = rhs(&u3, t + C3 * dt);
    let u4 = state.clone() * 0.178_079_954_393_132
        + u3.clone() * 0.821_920_045_606_868
        + l3.clone() * (0.544_974_750_228_521 * dt);

    // Final combination
    let l4 = rhs(&u4, t + C4 * dt);
    u2 * 0.517_231_671_970_585
        + u3 * 0.096_059_710_526_147
        + l3 * (0.063_692_468_666_290 * dt)
        + u4 * 0.386_708_617_503_269
        + l4 * (0.226_007_483_236_906 * dt)
}
