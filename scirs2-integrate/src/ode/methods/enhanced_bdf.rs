//! Enhanced BDF method with improved Jacobian handling
//!
//! This module provides an enhanced version of the Backward Differentiation Formula (BDF)
//! method for ODE solving with improved Jacobian handling, better error estimation,
//! and more efficient linear solvers.

use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::types::{ODEMethod, ODEOptions, ODEResult};
use crate::ode::utils::common::{calculate_error_weights, extrapolate};
use crate::ode::utils::jacobian::{
    newton_solve, JacobianManager, JacobianStrategy, JacobianStructure, NewtonParameters,
};
use crate::IntegrateFloat;
use ndarray::{Array1, ArrayView1};

/// Solve ODE using an enhanced Backward Differentiation Formula (BDF) method
///
/// This implementation features:
/// - Improved Jacobian handling with reuse, quasi-updates, and coloring
/// - Better error estimation for both step size and order control
/// - More efficient linear solvers for different Jacobian structures
/// - Comprehensive diagnostic information
///
/// # Arguments
///
/// * `f` - ODE function dy/dt = f(t, y)
/// * `t_span` - Time span [t_start, t_end]
/// * `y0` - Initial condition
/// * `opts` - Solver options
///
/// # Returns
///
/// The solution as an ODEResult or an error
pub fn enhanced_bdf_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Check BDF order is valid (1-5 supported)
    let order = opts.max_order.unwrap_or(2);
    if !(1..=5).contains(&order) {
        return Err(IntegrateError::ValueError(
            "BDF order must be between 1 and 5".to_string(),
        ));
    }

    // Initialize
    let [t_start, t_end] = t_span;
    let n_dim = y0.len();

    // Determine initial step size if not provided
    let h0 = opts.h0.unwrap_or_else(|| {
        // Simple heuristic for initial step size
        let span = t_end - t_start;
        span / F::from_usize(100).unwrap() * F::from_f64(0.1).unwrap() // 0.1% of interval
    });

    // Determine minimum and maximum step sizes
    let min_step = opts.min_step.unwrap_or_else(|| {
        let span = t_end - t_start;
        span * F::from_f64(1e-10).unwrap() // Minimal step size
    });

    let max_step = opts.max_step.unwrap_or_else(|| {
        t_end - t_start // Maximum step can be the whole interval
    });

    // Create the Jacobian manager with an appropriate strategy
    let jacobian_strategy = if n_dim <= 5 {
        JacobianStrategy::FiniteDifference // For small systems, full finite difference is fine
    } else if n_dim <= 100 {
        JacobianStrategy::BroydenUpdate // For medium systems, Broyden updates are efficient
    } else {
        JacobianStrategy::ModifiedNewton // For large systems, reuse Jacobian as much as possible
    };

    // Create Jacobian manager
    let jacobian_structure = if opts.use_banded_jacobian {
        let lower = opts.ml.unwrap_or(n_dim);
        let upper = opts.mu.unwrap_or(n_dim);
        JacobianStructure::Banded { lower, upper }
    } else {
        JacobianStructure::Dense
    };

    let mut jac_manager = JacobianManager::with_strategy(jacobian_strategy, jacobian_structure);

    // For BDF methods we need previous steps
    // We'll use another method (RK4) to bootstrap the initial steps
    // We need order previous points for a BDF method of order 'order'

    // First run RK4 to get the first 'order' points
    let mut h = h0;
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];
    let mut func_evals = 0;
    let mut step_count = 0;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;
    let mut newton_iters = F::zero();
    let mut n_lu = 0;
    let mut n_jac = 0;

    // Generate initial points using RK4 (more accurate than Euler)
    if order > 1 {
        let two = F::from_f64(2.0).unwrap();
        let mut t = t_start;
        let mut y = y0.clone();

        for _ in 0..(order - 1) {
            // Don't go beyond t_end
            if t + h > t_end {
                h = t_end - t;
            }

            // Standard RK4 step
            let half_step = h / two;

            let k1 = f(t, y.view());
            let k2 = f(t + half_step, (y.clone() + k1.clone() * half_step).view());
            let k3 = f(t + half_step, (y.clone() + k2.clone() * half_step).view());
            let k4 = f(t + h, (y.clone() + k3.clone() * h).view());
            func_evals += 4;

            // Combine slopes with appropriate weights
            let slope = (k1 + k2.clone() * two + k3.clone() * two + k4) / F::from_f64(6.0).unwrap();
            y = y + slope * h;

            // Update time
            t += h;

            // Store results
            t_values.push(t);
            y_values.push(y.clone());

            step_count += 1;
            accepted_steps += 1;

            // Don't continue if we've reached t_end
            if t >= t_end {
                break;
            }
        }
    }

    // Now we have enough points to start BDF
    let mut t = *t_values.last().unwrap();
    let mut y = y_values.last().unwrap().clone();

    // BDF coefficients for different orders
    // These are the coefficients for the BDF formula
    // For BDF of order p, we use p previous points

    // Coefficients for BDF1 (Implicit Euler) through BDF5
    let bdf_coefs: [Vec<F>; 5] = [
        // BDF1 (Implicit Euler): y_{n+1} - y_n = h * f(t_{n+1}, y_{n+1})
        vec![F::one(), F::from_f64(-1.0).unwrap()],
        // BDF2: 3/2 * y_{n+1} - 2 * y_n + 1/2 * y_{n-1} = h * f(t_{n+1}, y_{n+1})
        vec![
            F::from_f64(3.0 / 2.0).unwrap(),
            F::from_f64(-2.0).unwrap(),
            F::from_f64(1.0 / 2.0).unwrap(),
        ],
        // BDF3
        vec![
            F::from_f64(11.0 / 6.0).unwrap(),
            F::from_f64(-3.0).unwrap(),
            F::from_f64(3.0 / 2.0).unwrap(),
            F::from_f64(-1.0 / 3.0).unwrap(),
        ],
        // BDF4
        vec![
            F::from_f64(25.0 / 12.0).unwrap(),
            F::from_f64(-4.0).unwrap(),
            F::from_f64(3.0).unwrap(),
            F::from_f64(-4.0 / 3.0).unwrap(),
            F::from_f64(1.0 / 4.0).unwrap(),
        ],
        // BDF5
        vec![
            F::from_f64(137.0 / 60.0).unwrap(),
            F::from_f64(-5.0).unwrap(),
            F::from_f64(5.0).unwrap(),
            F::from_f64(-10.0 / 3.0).unwrap(),
            F::from_f64(5.0 / 4.0).unwrap(),
            F::from_f64(-1.0 / 5.0).unwrap(),
        ],
    ];

    // Initialize current order to the requested order (or lower if not enough history points)
    let mut current_order = order.min(t_values.len());
    let mut last_order_change = 0;
    let min_steps_before_order_change = 5;

    // Main integration loop
    while t < t_end && step_count < opts.max_steps {
        // Adjust step size for the last step if needed
        if t + h > t_end {
            h = t_end - t;
        }

        // Limit step size to bounds
        h = h.min(max_step).max(min_step);

        // Get coefficients for current order
        let coeffs = &bdf_coefs[current_order - 1];

        // Next time point
        let next_t = t + h;

        // Create historical times and values arrays for extrapolation
        let mut hist_times = Vec::with_capacity(t_values.len());
        let mut hist_values = Vec::with_capacity(y_values.len());

        // Collect the required history points based on current order
        let history_start = y_values.len().saturating_sub(current_order);
        for i in history_start..y_values.len() {
            hist_times.push(t_values[i]);
            hist_values.push(y_values[i].clone());
        }

        // Predict initial value using extrapolation from previous points
        let y_pred = if hist_values.len() > 1 {
            extrapolate(&hist_times, &hist_values, next_t)
        } else {
            y.clone()
        };

        // Create a function evaluations counter as a Cell to allow mutation
        let func_evals_cell = std::cell::Cell::new(0usize);

        // Create the nonlinear system for BDF
        let bdf_system = |y_next: &Array1<F>| {
            // Compute BDF residual:
            // c_0 * y_{n+1} - h * f(t_{n+1}, y_{n+1}) - sum_{j=1}^p c_j * y_{n+1-j} = 0

            // Evaluate function at the current iterate
            let f_eval = f(next_t, y_next.view());
            func_evals_cell.set(func_evals_cell.get() + 1);

            // Initialize residual with c_0 * y_{n+1} term
            let mut residual = y_next.clone() * coeffs[0];

            // Subtract previous values contribution
            for (j, coeff) in coeffs.iter().enumerate().skip(1).take(current_order) {
                if j <= y_values.len() {
                    let idx = y_values.len() - j;
                    residual = residual - y_values[idx].clone() * *coeff;
                }
            }

            // Subtract h * f(t_{n+1}, y_{n+1}) term
            residual = residual - f_eval.clone() * h;

            residual
        };

        // Update func_evals after the closure is done
        let prev_func_evals = func_evals;

        // Set up Newton solver parameters based on Jacobian strategy
        let update_freq = match jacobian_strategy {
            JacobianStrategy::FiniteDifference => 1, // Update every iteration
            JacobianStrategy::BroydenUpdate => 1,    // Update every iteration with Broyden
            JacobianStrategy::ModifiedNewton => 5,   // Update less frequently
            _ => 3,                                  // Default
        };

        let newton_params = NewtonParameters {
            max_iterations: 10,
            abs_tolerance: opts.rtol * F::from_f64(0.1).unwrap(),
            rel_tolerance: opts.rtol,
            jacobian_update_freq: update_freq,
            damping_factor: F::one(),
            min_damping: F::from_f64(0.1).unwrap(),
            force_jacobian_init: step_count == 0, // Force update on first step
            ..Default::default()
        };

        // Solve the nonlinear system using Newton's method
        let newton_result =
            newton_solve(bdf_system, y_pred.clone(), &mut jac_manager, newton_params);

        match newton_result {
            Ok(result) => {
                // Update counters including Cell-based func_evals
                func_evals = prev_func_evals + func_evals_cell.get() + result.func_evals;
                n_jac += result.jac_evals;
                n_lu += result.linear_solves;
                newton_iters += F::from(result.iterations).unwrap();

                // Update state
                let y_next = result.solution;

                // Compute error estimate by comparing with lower order solution
                // This is a more accurate error estimation strategy
                let error = if current_order > 1 {
                    // Compute error using solution from method one order lower
                    let lower_order = current_order - 1;
                    let lower_coeffs = &bdf_coefs[lower_order - 1];

                    // Compute the lower order solution using the values we already have
                    let mut rhs = Array1::<F>::zeros(n_dim);
                    for (j, &coeff) in lower_coeffs.iter().enumerate().skip(1).take(lower_order) {
                        if j <= y_values.len() {
                            let idx = y_values.len() - j;
                            rhs = rhs + y_values[idx].clone() * coeff;
                        }
                    }

                    // Add h * f term
                    let f_next = f(next_t, y_next.view());
                    func_evals += 1;
                    rhs = rhs + f_next.clone() * h;

                    // Solve for lower order solution
                    let mut y_lower = Array1::<F>::zeros(n_dim);
                    for i in 0..n_dim {
                        y_lower[i] = rhs[i] / lower_coeffs[0];
                    }

                    // Compute scaled error between higher and lower order solutions
                    let tol_scale = calculate_error_weights(&y_next, opts.atol, opts.rtol);
                    let mut max_err = F::zero();
                    for i in 0..n_dim {
                        let err = (y_next[i] - y_lower[i]).abs() / tol_scale[i];
                        max_err = max_err.max(err);
                    }
                    max_err
                } else {
                    // For order 1, we can't compare with a lower order
                    // Use a simpler error estimate based on residual
                    result.error_estimate
                };

                // Step size adjustment based on error
                let err_order = F::from_usize(current_order + 1).unwrap();
                let mut factor = if error > F::zero() {
                    F::from_f64(0.9).unwrap() * (F::one() / error).powf(F::one() / err_order)
                } else {
                    F::from_f64(5.0).unwrap() // Maximum increase if error is zero
                };

                // Apply safety factors
                let factor_max = F::from_f64(5.0).unwrap();
                let factor_min = F::from_f64(0.2).unwrap();
                factor = factor.min(factor_max).max(factor_min);

                // Check if step is acceptable
                if error <= F::one() {
                    // Step accepted
                    t = next_t;
                    y = y_next;

                    // Store results
                    t_values.push(t);
                    y_values.push(y.clone());

                    step_count += 1;
                    accepted_steps += 1;

                    // Adjust step size for next step
                    h *= factor;

                    // Consider order adjustment - but not too frequently
                    if step_count - last_order_change >= min_steps_before_order_change {
                        // Strategy: try to estimate the error of the current order and one order higher
                        if current_order < order
                            && error < opts.rtol
                            && y_values.len() > current_order
                        {
                            // Consider increasing order
                            current_order += 1;
                            last_order_change = step_count;
                        } else if current_order > 1
                            && (error > F::from_f64(0.5).unwrap() || result.iterations > 8)
                        {
                            // Consider decreasing order
                            current_order -= 1;
                            last_order_change = step_count;
                        }
                    }
                } else {
                    // Step rejected
                    h *= factor;
                    rejected_steps += 1;

                    // If we've reduced step size too much, report error
                    if h < min_step {
                        return Err(IntegrateError::ConvergenceError(
                            "Step size too small after rejection".to_string(),
                        ));
                    }
                }
            }
            Err(e) => {
                // Newton failed to converge
                h *= F::from_f64(0.5).unwrap();
                rejected_steps += 1;

                // If step size is too small, return error
                if h < min_step {
                    return Err(e);
                }
            }
        }
    }

    let success = t >= t_end;
    let message = if !success {
        Some(format!(
            "Maximum number of steps ({}) reached",
            opts.max_steps
        ))
    } else {
        Some(format!(
            "Integration completed successfully. Jacobian strategy: {:?}, Final order: {}",
            jacobian_strategy, current_order
        ))
    };

    // Return the solution
    Ok(ODEResult {
        t: t_values,
        y: y_values,
        success,
        message,
        n_eval: func_evals,
        n_steps: step_count,
        n_accepted: accepted_steps,
        n_rejected: rejected_steps,
        n_lu,
        n_jac,
        method: ODEMethod::Bdf,
    })
}
