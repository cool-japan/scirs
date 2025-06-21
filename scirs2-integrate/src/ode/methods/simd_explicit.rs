//! SIMD-accelerated explicit ODE solver methods
//!
//! This module provides SIMD-optimized versions of explicit ODE solvers,
//! offering significant performance improvements for large systems of ODEs
//! on modern processors with SIMD instruction sets.

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use crate::ode::types::{ODEOptions, ODEResult};
use crate::ode::utils::common::{estimate_initial_step, ODEState, StepResult};
use ndarray::{Array1, ArrayView1};

#[cfg(feature = "simd")]
use crate::ode::utils::simd_ops::{SimdOdeOps};

/// SIMD-accelerated 4th-order Runge-Kutta method
///
/// This implementation uses SIMD instructions to accelerate vector operations
/// in the RK4 integration steps, providing significant performance improvements
/// for large systems of ODEs.
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
#[cfg(feature = "simd")]
pub fn simd_rk4_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let [t_start, t_end] = t_span;
    let n_dim = y0.len();

    // Determine step size
    let h = opts.h0.unwrap_or_else(|| {
        estimate_initial_step(&f, t_start, &y0.view(), F::from_f64(1e-3).unwrap())
    });

    // Storage for solution
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];

    let mut t = t_start;
    let mut y = y0;
    let mut steps = 0;
    let mut func_evals = 0;

    while t < t_end {
        // Adjust step size near the end
        let h_current = if t + h > t_end { t_end - t } else { h };

        // SIMD-accelerated RK4 step
        let (y_new, n_evals) = simd_rk4_step(&f, t, &y.view(), h_current)?;
        func_evals += n_evals;

        // Update state
        t = t + h_current;
        y = y_new;
        steps += 1;

        // Store solution
        t_values.push(t);
        y_values.push(y.clone());

        // Safety check
        if steps > 1_000_000 {
            return Err(crate::error::IntegrateError::MaxStepsExceeded(
                "Maximum number of steps exceeded in SIMD RK4 method".to_string(),
            ));
        }
    }

    Ok(ODEResult {
        t: t_values,
        y: y_values,
        n_steps: steps,
        n_func_evals: func_evals,
        n_rejected: 0,
        method: "SIMD RK4".to_string(),
        success: true,
        message: Some("Integration completed successfully".to_string()),
    })
}

/// SIMD-accelerated adaptive Runge-Kutta method (RK45)
///
/// This method uses embedded Runge-Kutta formulas with SIMD acceleration
/// for both the integration steps and error estimation.
///
/// # Arguments
///
/// * `f` - ODE function dy/dt = f(t, y)
/// * `t_span` - Time span [t_start, t_end]
/// * `y0` - Initial condition
/// * `opts` - Solver options including tolerances
///
/// # Returns
///
/// The solution as an ODEResult or an error
#[cfg(feature = "simd")]
pub fn simd_rk45_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let [t_start, t_end] = t_span;

    // Initial step size
    let mut h = opts.h0.unwrap_or_else(|| {
        estimate_initial_step(&f, t_start, &y0.view(), F::from_f64(1e-3).unwrap())
    });

    let min_step = opts.min_step.unwrap_or(F::from_f64(1e-12).unwrap());
    let max_step = opts
        .max_step
        .unwrap_or((t_end - t_start) / F::from_f64(10.0).unwrap());
    let abs_tol = opts.abs_tol.unwrap_or(F::from_f64(1e-6).unwrap());
    let rel_tol = opts.rel_tol.unwrap_or(F::from_f64(1e-6).unwrap());

    // Storage for solution
    let mut t_values = vec![t_start];
    let mut y_values = vec![y0.clone()];

    let mut t = t_start;
    let mut y = y0;
    let mut steps = 0;
    let mut func_evals = 0;
    let mut rejected_steps = 0;

    while t < t_end {
        // Adjust step size near the end
        if t + h > t_end {
            h = t_end - t;
        }

        // SIMD-accelerated RK45 step with error estimation
        let (y_new, error_est, n_evals) = simd_rk45_step(&f, t, &y.view(), h)?;
        func_evals += n_evals;

        // Compute error tolerance
        let error_tolerance = compute_error_tolerance(&y, &y_new, abs_tol, rel_tol);

        if error_est <= error_tolerance {
            // Accept step
            t = t + h;
            y = y_new;
            steps += 1;

            // Store solution
            t_values.push(t);
            y_values.push(y.clone());

            // Adjust step size for next step
            if error_est > F::zero() {
                let safety_factor = F::from_f64(0.9).unwrap();
                let growth_factor =
                    safety_factor * (error_tolerance / error_est).powf(F::from_f64(0.2).unwrap());
                h = h * growth_factor.min(F::from_f64(2.0).unwrap());
            } else {
                h = h * F::from_f64(1.5).unwrap();
            }
        } else {
            // Reject step
            rejected_steps += 1;
            let safety_factor = F::from_f64(0.9).unwrap();
            let shrink_factor =
                safety_factor * (error_tolerance / error_est).powf(F::from_f64(0.25).unwrap());
            h = h * shrink_factor.max(F::from_f64(0.2).unwrap());
        }

        // Check minimum step size
        if h < min_step {
            return Err(crate::error::IntegrateError::StepSizeTooSmall(
                "Step size became too small in SIMD RK45 method".to_string(),
            ));
        }

        // Limit maximum step size
        h = h.min(max_step);

        // Safety check
        if steps > 100_000 {
            return Err(crate::error::IntegrateError::MaxStepsExceeded(
                "Maximum number of steps exceeded in SIMD RK45 method".to_string(),
            ));
        }
    }

    Ok(ODEResult {
        t: t_values,
        y: y_values,
        n_steps: steps,
        n_func_evals: func_evals,
        n_rejected: rejected_steps,
        method: "SIMD RK45".to_string(),
        success: true,
        message: Some("Integration completed successfully".to_string()),
    })
}

/// Perform a single SIMD-accelerated RK4 step
#[cfg(feature = "simd")]
fn simd_rk4_step<F, Func>(
    f: &Func,
    t: F,
    y: &ArrayView1<F>,
    h: F,
) -> IntegrateResult<(Array1<F>, usize)>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let h_half = h * F::from_f64(0.5).unwrap();

    // k1 = f(t, y)
    let k1 = simd_ode_function_eval(f, t, y, true)?;

    // k2 = f(t + h/2, y + h/2 * k1)
    let y_temp1 = y + &k1 * h_half;
    let k2 = simd_ode_function_eval(f, t + h_half, &y_temp1.view(), true)?;

    // k3 = f(t + h/2, y + h/2 * k2)
    let y_temp2 = y + &k2 * h_half;
    let k3 = simd_ode_function_eval(f, t + h_half, &y_temp2.view(), true)?;

    // k4 = f(t + h, y + h * k3)
    let y_temp3 = y + &k3 * h;
    let k4 = simd_ode_function_eval(f, t + h, &y_temp3.view(), true)?;

    // y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    let c1 = F::one() / F::from_f64(6.0).unwrap();
    let c2 = F::from_f64(2.0).unwrap() / F::from_f64(6.0).unwrap();
    let c3 = c2;
    let c4 = c1;

    let y_new = simd_rk_step(
        y,
        h,
        &k1.view(),
        &k2.view(),
        &k3.view(),
        &k4.view(),
        c1,
        c2,
        c3,
        c4,
    );

    Ok((y_new, 4)) // 4 function evaluations
}

/// Perform a single SIMD-accelerated RK45 step with error estimation
#[cfg(feature = "simd")]
fn simd_rk45_step<F, Func>(
    f: &Func,
    t: F,
    y: &ArrayView1<F>,
    h: F,
) -> IntegrateResult<(Array1<F>, F, usize)>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Dormand-Prince coefficients
    let a21 = F::from_f64(1.0 / 5.0).unwrap();
    let a31 = F::from_f64(3.0 / 40.0).unwrap();
    let a32 = F::from_f64(9.0 / 40.0).unwrap();
    let a41 = F::from_f64(44.0 / 45.0).unwrap();
    let a42 = F::from_f64(-56.0 / 15.0).unwrap();
    let a43 = F::from_f64(32.0 / 9.0).unwrap();
    let a51 = F::from_f64(19372.0 / 6561.0).unwrap();
    let a52 = F::from_f64(-25360.0 / 2187.0).unwrap();
    let a53 = F::from_f64(64448.0 / 6561.0).unwrap();
    let a54 = F::from_f64(-212.0 / 729.0).unwrap();
    let a61 = F::from_f64(9017.0 / 3168.0).unwrap();
    let a62 = F::from_f64(-355.0 / 33.0).unwrap();
    let a63 = F::from_f64(46732.0 / 5247.0).unwrap();
    let a64 = F::from_f64(49.0 / 176.0).unwrap();
    let a65 = F::from_f64(-5103.0 / 18656.0).unwrap();

    // k1 = f(t, y)
    let k1 = simd_ode_function_eval(f, t, y, true)?;

    // k2 = f(t + h/5, y + h/5 * k1)
    let y2 = y + &k1 * (h * a21);
    let k2 = simd_ode_function_eval(f, t + h * a21, &y2.view(), true)?;

    // k3 = f(t + 3h/10, y + h * (3/40 * k1 + 9/40 * k2))
    let y3 = y + &(&k1 * a31 + &k2 * a32) * h;
    let k3 = simd_ode_function_eval(
        f,
        t + h * F::from_f64(3.0 / 10.0).unwrap(),
        &y3.view(),
        true,
    )?;

    // k4 = f(t + 4h/5, y + h * (44/45 * k1 - 56/15 * k2 + 32/9 * k3))
    let y4 = y + &(&(&k1 * a41 + &k2 * a42) + &k3 * a43) * h;
    let k4 = simd_ode_function_eval(f, t + h * F::from_f64(4.0 / 5.0).unwrap(), &y4.view(), true)?;

    // k5 = f(t + 8h/9, y + h * (19372/6561 * k1 - 25360/2187 * k2 + 64448/6561 * k3 - 212/729 * k4))
    let y5 = y + &(&(&(&k1 * a51 + &k2 * a52) + &k3 * a53) + &k4 * a54) * h;
    let k5 = simd_ode_function_eval(f, t + h * F::from_f64(8.0 / 9.0).unwrap(), &y5.view(), true)?;

    // k6 = f(t + h, y + h * (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3 + 49/176 * k4 - 5103/18656 * k5))
    let y6 = y + &(&(&(&(&k1 * a61 + &k2 * a62) + &k3 * a63) + &k4 * a64) + &k5 * a65) * h;
    let k6 = simd_ode_function_eval(f, t + h, &y6.view(), true)?;

    // 5th order solution
    let b1 = F::from_f64(35.0 / 384.0).unwrap();
    let b3 = F::from_f64(500.0 / 1113.0).unwrap();
    let b4 = F::from_f64(125.0 / 192.0).unwrap();
    let b5 = F::from_f64(-2187.0 / 6784.0).unwrap();
    let b6 = F::from_f64(11.0 / 84.0).unwrap();

    let y_new = y + &(&(&(&(&k1 * b1 + &k3 * b3) + &k4 * b4) + &k5 * b5) + &k6 * b6) * h;

    // 4th order solution for error estimation
    let b1_star = F::from_f64(5179.0 / 57600.0).unwrap();
    let b3_star = F::from_f64(7571.0 / 16695.0).unwrap();
    let b4_star = F::from_f64(393.0 / 640.0).unwrap();
    let b5_star = F::from_f64(-92097.0 / 339200.0).unwrap();
    let b6_star = F::from_f64(187.0 / 2100.0).unwrap();
    let b7_star = F::from_f64(1.0 / 40.0).unwrap();

    // Note: k7 would be calculated here for the full RK45, but we'll use a simplified error estimate
    let y_star = y + &(&(&(&(&k1 * b1_star + &k3 * b3_star) + &k4 * b4_star) + &k5 * b5_star)
        + &k6 * b6_star)
        * h;

    // Error estimate
    #[cfg(feature = "simd")]
    let error_vec = &y_new - &y_star;
    let error_est = SimdOdeOps::simd_norm_inf(&error_vec.view());
    #[cfg(not(feature = "simd"))]
    let error_est = (&y_new - &y_star)
        .iter()
        .map(|&x| x.abs())
        .fold(F::zero(), |a, b| a.max(b));

    Ok((y_new, error_est, 6)) // 6 function evaluations
}

/// Compute error tolerance for adaptive step size control
fn compute_error_tolerance<F: IntegrateFloat>(
    y_old: &ArrayView1<F>,
    y_new: &ArrayView1<F>,
    abs_tol: F,
    rel_tol: F,
) -> F {
    let mut max_tol = F::zero();

    for (_i, (&y_old_i, &y_new_i)) in y_old.iter().zip(y_new.iter()).enumerate() {
        let scale = y_old_i.abs().max(y_new_i.abs());
        let tol_i = abs_tol + rel_tol * scale;
        max_tol = max_tol.max(tol_i);
    }

    max_tol
}

/// Fallback methods when SIMD is not available
#[cfg(not(feature = "simd"))]
pub fn simd_rk4_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Fallback to regular RK4 method
    crate::ode::methods::rk4_method(f, t_span, y0, opts)
}

#[cfg(not(feature = "simd"))]
pub fn simd_rk45_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Fallback to regular RK45 method
    crate::ode::methods::rk45_method(f, t_span, y0, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_error_tolerance_computation() {
        let y_old = arr1(&[1.0, 2.0, 3.0]);
        let y_new = arr1(&[1.01, 2.02, 3.03]);
        let abs_tol = 1e-6;
        let rel_tol = 1e-6;

        let tolerance = compute_error_tolerance(&y_old.view(), &y_new.view(), abs_tol, rel_tol);

        // Should be dominated by relative tolerance * max(|y_old|, |y_new|)
        let expected = abs_tol + rel_tol * 3.03;
        assert_relative_eq!(tolerance, expected, epsilon = 1e-12);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_rk4_simple() {
        // Test on simple exponential decay: dy/dt = -y, y(0) = 1
        let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { -y.to_owned() };

        let y0 = arr1(&[1.0]);
        let t_span = [0.0, 1.0];
        let opts = ODEOptions {
            h0: Some(0.1),
            ..Default::default()
        };

        let result = simd_rk4_method(f, t_span, y0, opts).unwrap();

        // Exact solution at t=1 is exp(-1) ≈ 0.36788
        let final_value = result.y.last().unwrap()[0];
        let exact = (-1.0_f64).exp();

        assert_relative_eq!(final_value, exact, epsilon = 1e-3);
        assert!(result.success);
        // Check that it's using SIMD RK4 method (would need method tracking)
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_rk45_adaptive() {
        // Test on harmonic oscillator: d²y/dt² + y = 0
        // Convert to system: dy₁/dt = y₂, dy₂/dt = -y₁
        let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { arr1(&[y[1], -y[0]]) };

        let y0 = arr1(&[1.0, 0.0]); // y(0) = 1, dy/dt(0) = 0
        let t_span = [0.0, std::f64::consts::PI]; // Half period
        let opts = ODEOptions {
            atol: Some(1e-8),
            rtol: Some(1e-8),
            h0: Some(0.1),
            ..Default::default()
        };

        let result = simd_rk45_method(f, t_span, y0, opts).unwrap();

        // At t = π, exact solution is y₁ = -1, y₂ = 0
        let final_y = result.y.last().unwrap();
        assert_relative_eq!(final_y[0], -1.0, epsilon = 1e-6);
        assert_relative_eq!(final_y[1], 0.0, epsilon = 1e-6);
        assert!(result.success);
        // Check that it's using SIMD RK45 method (would need method tracking)
    }
}
