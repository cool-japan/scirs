//! Explicit ODE solver methods
//!
//! This module implements explicit methods for solving ODEs,
//! including Euler's method and the classic 4th-order Runge-Kutta method.

use crate::error::IntegrateResult;
use crate::ode::types::{ODEMethod, ODEOptions, ODEResult};
use crate::IntegrateFloat;
use ndarray::{Array1, ArrayView1};

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
    let two = F::from_f64(2.0).unwrap();
    let six = F::from_f64(6.0).unwrap();

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
