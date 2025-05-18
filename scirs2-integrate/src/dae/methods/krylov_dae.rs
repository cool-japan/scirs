//! Krylov Subspace Methods for Large DAE Systems
//!
//! This module provides Krylov subspace-based solvers for large differential algebraic
//! equation (DAE) systems. These methods are particularly effective for large, sparse
//! DAE systems where direct linear solvers become inefficient.
//!
//! The implementation uses iterative Krylov methods (like GMRES) to solve the linear
//! systems that arise in the Newton iterations for implicit DAE solvers.

use crate::common::IntegrateFloat;
use crate::dae::types::{DAEIndex, DAEOptions, DAEResult, DAEType};
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::ODEMethod;
use ndarray::{Array1, Array2, ArrayView1};

/// Type alias for GMRES result
type GMRESResult<F> = (Array2<F>, Vec<Array1<F>>, F, usize);

/// Maximum number of GMRES iterations
#[allow(dead_code)]
const MAX_GMRES_ITER: usize = 100;

/// GMRES restart parameter
const GMRES_RESTART: usize = 30;

/// Default GMRES tolerance
const GMRES_TOL: f64 = 1e-8;

/// Krylov-enhanced BDF method for semi-explicit DAE systems
///
/// Implements a specialized BDF method for semi-explicit DAE systems that
/// uses Krylov subspace methods (GMRES) to solve the linear systems that
/// arise in the Newton iterations. This approach is particularly effective
/// for large, sparse DAE systems.
///
/// The semi-explicit form is:
/// x' = f(x, y, t)
/// 0 = g(x, y, t)
pub fn krylov_bdf_semi_explicit_dae<F, FFunc, GFunc>(
    f: FFunc,
    g: GFunc,
    t_span: [F; 2],
    x0: Array1<F>,
    y0: Array1<F>,
    options: DAEOptions<F>,
) -> IntegrateResult<DAEResult<F>>
where
    F: IntegrateFloat,
    FFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
    GFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
{
    // Get dimensions
    let n_x = x0.len();
    let n_y = y0.len();
    let n_total = n_x + n_y;

    // Storage for solution
    let mut t_values = vec![t_span[0]];
    let mut x_values = vec![x0.clone()];
    let mut y_values = vec![y0.clone()];

    // Current values
    let mut t_current = t_span[0];
    let mut _x_current = x0.clone();
    let mut _y_current = y0.clone();

    // Initial step size
    let mut h = options.h0.unwrap_or_else(|| {
        let span = t_span[1] - t_span[0];
        span * F::from_f64(0.01).unwrap() // 1% of interval
    });

    // Step limits
    let min_step = options.min_step.unwrap_or_else(|| {
        let span = t_span[1] - t_span[0];
        span * F::from_f64(1e-6).unwrap() // Very small relative to interval
    });

    let max_step = options.max_step.unwrap_or_else(|| {
        let span = t_span[1] - t_span[0];
        span * F::from_f64(0.1).unwrap() // 10% of interval
    });

    // Maximum BDF order
    let max_order = options.max_order.unwrap_or(5).min(5);

    // Tolerances
    let rtol = options.rtol;
    let atol = options.atol;

    // Krylov method settings
    let gmres_tol = F::from_f64(GMRES_TOL).unwrap();
    let max_gmres_iter = options.max_newton_iterations * 3; // Allow more GMRES iterations than Newton

    // Counters for statistics
    let mut n_steps = 0;
    let mut n_accepted = 0;
    let mut n_rejected = 0;
    let mut n_f_evals = 0;
    let mut n_g_evals = 0;
    let mut n_jac_evals = 0;
    let mut n_krylov_iters = 0;

    // BDF method coefficients for various orders
    // These are the coefficients in the BDF formula:
    // Σ α_j * y_{n+1-j} = h * β * f(t_{n+1}, y_{n+1})
    // where α_0 = 1 for all orders

    // Alpha coefficients (exclude α_0 = 1)
    let alpha_coeffs = [
        // Order 1 (Backward Euler)
        vec![F::from_f64(-1.0).unwrap()],
        // Order 2
        vec![
            F::from_f64(-4.0 / 3.0).unwrap(),
            F::from_f64(1.0 / 3.0).unwrap(),
        ],
        // Order 3
        vec![
            F::from_f64(-18.0 / 11.0).unwrap(),
            F::from_f64(9.0 / 11.0).unwrap(),
            F::from_f64(-2.0 / 11.0).unwrap(),
        ],
        // Order 4
        vec![
            F::from_f64(-48.0 / 25.0).unwrap(),
            F::from_f64(36.0 / 25.0).unwrap(),
            F::from_f64(-16.0 / 25.0).unwrap(),
            F::from_f64(3.0 / 25.0).unwrap(),
        ],
        // Order 5
        vec![
            F::from_f64(-300.0 / 137.0).unwrap(),
            F::from_f64(300.0 / 137.0).unwrap(),
            F::from_f64(-200.0 / 137.0).unwrap(),
            F::from_f64(75.0 / 137.0).unwrap(),
            F::from_f64(-12.0 / 137.0).unwrap(),
        ],
    ];

    // Beta coefficients (multiplier for the RHS function)
    let beta_coeffs = [
        F::from_f64(1.0).unwrap(),          // Order 1
        F::from_f64(2.0 / 3.0).unwrap(),    // Order 2
        F::from_f64(6.0 / 11.0).unwrap(),   // Order 3
        F::from_f64(12.0 / 25.0).unwrap(),  // Order 4
        F::from_f64(60.0 / 137.0).unwrap(), // Order 5
    ];

    // Current order (start with order 1 and build up)
    let mut order = 1;

    // Main integration loop
    while t_current < t_span[1] && n_steps < options.max_steps {
        // Adjust step size to hit the end point exactly if needed
        if t_current + h > t_span[1] {
            h = t_span[1] - t_current;
        }

        // New time point
        let t_new = t_current + h;

        // Predict step: extrapolate from previous points
        let (x_pred, y_pred) = predict_step(&x_values, &y_values, order, h);

        // Store original predictions for error estimation
        let x_pred_orig = x_pred.clone();
        let y_pred_orig = y_pred.clone();

        // Evaluate constraint at the predictor point
        let _g_pred = g(t_new, x_pred.view(), y_pred.view());
        n_g_evals += 1;

        // Get the alpha coefficient for this order
        let alpha = alpha_coeffs[order - 1].clone();
        let beta = beta_coeffs[order - 1];

        // Create a history array for x
        let mut x_history = Vec::with_capacity(order);
        for i in 0..order {
            let idx = x_values.len() - 1 - i;
            x_history.push(x_values[idx].clone());
        }

        // Initialize corrector iteration (Newton's method)
        let mut x_corr = x_pred.clone();
        let mut y_corr = y_pred.clone();

        // Maximum Newton iterations
        let max_newton_iter = options.max_newton_iterations;

        // Newton iteration for corrector
        let mut converged = false;
        for _iter in 0..max_newton_iter {
            // Evaluate f at the current corrector value
            let f_val = f(t_new, x_corr.view(), y_corr.view());
            n_f_evals += 1;

            // Evaluate g at the current corrector value
            let g_val = g(t_new, x_corr.view(), y_corr.view());
            n_g_evals += 1;

            // Compute the residuals for the BDF formula and constraint
            let mut residual_x = Array1::zeros(n_x);

            // Compute the BDF formula residual: x_corr - Σ α_j * x_{n-j} - h * β * f(t_new, x_corr, y_corr)
            for i in 0..n_x {
                // First term: x_corr
                residual_x[i] = x_corr[i];

                // Historical terms: - Σ α_j * x_{n-j}
                for j in 0..order {
                    residual_x[i] += alpha[j] * x_history[j][i];
                }

                // Function term: - h * β * f
                residual_x[i] -= h * beta * f_val[i];
            }

            // Constraint residual is simply g_val
            let residual_g = g_val;

            // Compute the full residual norm
            let res_x_norm = residual_x
                .iter()
                .fold(F::zero(), |acc, &val| acc + val * val)
                .sqrt();
            let res_g_norm = residual_g
                .iter()
                .fold(F::zero(), |acc, &val| acc + val * val)
                .sqrt();

            let residual_norm = (res_x_norm * res_x_norm + res_g_norm * res_g_norm).sqrt();

            // Check convergence
            if residual_norm < options.newton_tol {
                converged = true;
                break;
            }

            // Compute the Jacobian of f with respect to x and y
            let f_x = compute_jacobian_x(
                &f,
                t_new,
                x_corr.view(),
                y_corr.view(),
                F::from_f64(1e-8).unwrap(),
            );
            let f_y = compute_jacobian_y(
                &f,
                t_new,
                x_corr.view(),
                y_corr.view(),
                F::from_f64(1e-8).unwrap(),
            );
            n_jac_evals += 2;

            // Compute the Jacobian of g with respect to x and y
            let g_x = compute_jacobian_x(
                &g,
                t_new,
                x_corr.view(),
                y_corr.view(),
                F::from_f64(1e-8).unwrap(),
            );
            let g_y = compute_jacobian_y(
                &g,
                t_new,
                x_corr.view(),
                y_corr.view(),
                F::from_f64(1e-8).unwrap(),
            );
            n_jac_evals += 2;

            // Construct the full residual vector
            let mut full_residual = Array1::<F>::zeros(n_total);
            for i in 0..n_x {
                full_residual[i] = residual_x[i];
            }
            for i in 0..n_y {
                full_residual[n_x + i] = residual_g[i];
            }

            // Negate the residual for solving J * Δz = -residual
            let neg_residual = full_residual.mapv(|x| -x);

            // Create a matrix-vector product function for GMRES
            // This avoids explicitly forming the full Jacobian matrix
            let jacobian_matvec = |v: &Array1<F>| -> Array1<F> {
                let mut result = Array1::<F>::zeros(n_total);

                // Extract the x and y components of the input vector
                let v_x = v.slice(ndarray::s![0..n_x]).to_owned();
                let v_y = v.slice(ndarray::s![n_x..]).to_owned();

                // Compute the product for the x-x block: [I - h * β * ∂f/∂x] * v_x
                for i in 0..n_x {
                    result[i] = v_x[i]; // Identity part
                    for j in 0..n_x {
                        // Subtract h * β * ∂f/∂x * v_x
                        result[i] -= h * beta * f_x[[i, j]] * v_x[j];
                    }
                }

                // Compute the product for the x-y block: [-h * β * ∂f/∂y] * v_y
                for i in 0..n_x {
                    for j in 0..n_y {
                        // Subtract h * β * ∂f/∂y * v_y
                        result[i] -= h * beta * f_y[[i, j]] * v_y[j];
                    }
                }

                // Compute the product for the g-x block: [∂g/∂x] * v_x
                for i in 0..n_y {
                    for j in 0..n_x {
                        result[n_x + i] += g_x[[i, j]] * v_x[j];
                    }
                }

                // Compute the product for the g-y block: [∂g/∂y] * v_y
                for i in 0..n_y {
                    for j in 0..n_y {
                        result[n_x + i] += g_y[[i, j]] * v_y[j];
                    }
                }

                result
            };

            // Create a simple diagonal preconditioner
            let diag_preconditioner = |v: &Array1<F>| -> Array1<F> {
                // Extract diagonals of Jacobian parts
                let mut diag = Array1::<F>::ones(n_total);

                // For the x-x block diagonal: (1 - h * β * ∂f_i/∂x_i)
                for i in 0..n_x {
                    diag[i] = F::one() - h * beta * f_x[[i, i]];
                    if diag[i].abs() < F::from_f64(1e-14).unwrap() {
                        diag[i] = F::from_f64(1e-14).unwrap() * diag[i].signum();
                    }
                    diag[i] = F::one() / diag[i];
                }

                // For the g-y block diagonal: ∂g_i/∂y_i
                for i in 0..n_y {
                    diag[n_x + i] = g_y[[i, i]];
                    if diag[n_x + i].abs() < F::from_f64(1e-14).unwrap() {
                        diag[n_x + i] = F::from_f64(1e-14).unwrap() * diag[n_x + i].signum();
                    }
                    diag[n_x + i] = F::one() / diag[n_x + i];
                }

                // Apply diagonal preconditioner: P⁻¹v = D⁻¹v
                v.iter()
                    .zip(diag.iter())
                    .map(|(&vi, &di)| vi * di)
                    .collect()
            };

            // Solve the linear system using GMRES
            let (delta_z, gmres_iterations) = match gmres_solver(
                jacobian_matvec,
                &neg_residual,
                Some(diag_preconditioner),
                gmres_tol,
                max_gmres_iter,
                GMRES_RESTART,
            ) {
                Ok((dz, iters)) => (dz, iters),
                Err(_e) => {
                    // If the linear solve fails, try with a smaller step
                    // and terminate this Newton iteration
                    h *= F::from_f64(0.5).unwrap();
                    break;
                }
            };
            n_krylov_iters += gmres_iterations;

            // Extract the x and y components of the solution
            let delta_x = delta_z.slice(ndarray::s![0..n_x]).to_owned();
            let delta_y = delta_z.slice(ndarray::s![n_x..]).to_owned();

            // Apply the Newton step with damping if needed
            let mut alpha_damp = F::one();
            let min_alpha = F::from_f64(0.1).unwrap();

            // Damped Newton iteration to improve convergence
            while alpha_damp >= min_alpha {
                // Apply the damped step
                let x_new = &x_corr + &(&delta_x * alpha_damp);
                let y_new = &y_corr + &(&delta_y * alpha_damp);

                // Evaluate the residual at the new point
                let f_new = f(t_new, x_new.view(), y_new.view());
                let g_new = g(t_new, x_new.view(), y_new.view());
                n_f_evals += 1;
                n_g_evals += 1;

                // Compute the new residuals
                let mut residual_x_new = Array1::zeros(n_x);
                for i in 0..n_x {
                    residual_x_new[i] = x_new[i];
                    for j in 0..order {
                        residual_x_new[i] += alpha[j] * x_history[j][i];
                    }
                    residual_x_new[i] -= h * beta * f_new[i];
                }

                let res_x_new_norm = residual_x_new
                    .iter()
                    .fold(F::zero(), |acc, &val| acc + val * val)
                    .sqrt();
                let res_g_new_norm = g_new
                    .iter()
                    .fold(F::zero(), |acc, &val| acc + val * val)
                    .sqrt();

                let residual_new_norm =
                    (res_x_new_norm * res_x_new_norm + res_g_new_norm * res_g_new_norm).sqrt();

                // Accept if the residual is reduced
                if residual_new_norm < residual_norm {
                    x_corr = x_new;
                    y_corr = y_new;
                    break;
                }

                // Reduce damping factor
                alpha_damp *= F::from_f64(0.5).unwrap();
            }

            // If damping factor got too small, the Newton iteration is not converging
            if alpha_damp < min_alpha {
                // Reduce step size and try again
                h *= F::from_f64(0.5).unwrap();
                break;
            }
        }

        // Check for convergence of the Newton iteration
        if !converged {
            // If not converged, reduce step size and try again
            h *= F::from_f64(0.5).unwrap();

            // If step size gets too small, the problem might be too stiff
            if h < min_step {
                return Err(IntegrateError::ComputationError(format!(
                    "Failed to converge at t = {}. Step size too small.",
                    t_current
                )));
            }

            n_rejected += 1;
            continue;
        }

        // Step accepted, update step count
        n_steps += 1;
        n_accepted += 1;

        // Estimate local error for step size control
        // For BDF methods, we can use the difference between the predictor and corrector
        let error_x = (&x_corr - &x_pred_orig).mapv(|x| x.abs());
        let error_y = (&y_corr - &y_pred_orig).mapv(|x| x.abs());

        // Compute scaled error norm
        let mut error_norm_x = F::zero();
        for i in 0..n_x {
            let scale = atol + rtol * x_corr[i].abs();
            error_norm_x += (error_x[i] / scale).powi(2);
        }
        error_norm_x = (error_norm_x / F::from_usize(n_x).unwrap()).sqrt();

        let mut error_norm_y = F::zero();
        for i in 0..n_y {
            let scale = atol + rtol * y_corr[i].abs();
            error_norm_y += (error_y[i] / scale).powi(2);
        }
        error_norm_y = (error_norm_y / F::from_usize(n_y).unwrap()).sqrt();

        // Take the maximum of the two error norms
        let error_norm = error_norm_x.max(error_norm_y);

        // Adjust the step size based on the error estimate
        let error_order = order as i32;
        let safety = F::from_f64(0.9).unwrap(); // Safety factor

        // Calculate the optimal step size
        let h_new = if error_norm > F::zero() {
            h * safety * (F::one() / error_norm).powf(F::one() / F::from_i32(error_order).unwrap())
        } else {
            h * F::from_f64(2.0).unwrap() // Double the step size if error is 0
        };

        // Limit step size increase and decrease
        let max_increase = F::from_f64(2.0).unwrap();
        let max_decrease = F::from_f64(0.1).unwrap();

        let h_new = (h_new / h).max(max_decrease).min(max_increase) * h;
        let h_new = h_new.min(max_step).max(min_step);

        // Update step size for next iteration
        h = h_new;

        // Store the solution
        t_values.push(t_new);
        x_values.push(x_corr.clone());
        y_values.push(y_corr.clone());

        // Update current values
        t_current = t_new;
        _x_current = x_corr;
        _y_current = y_corr;

        // Adjust the order based on history
        if n_steps >= 5 {
            // At this point, we have enough history to consider increasing the order
            if order < max_order {
                order = (order + 1).min(max_order);
            }
        }
    }

    // Check if we reached the end time
    let success = t_current >= t_span[1];

    // Create result
    let result = DAEResult {
        t: t_values,
        x: x_values,
        y: y_values,
        success,
        message: if success {
            Some(format!(
                "Successful integration. {} steps taken, {} GMRES iterations.",
                n_steps, n_krylov_iters
            ))
        } else {
            Some(format!(
                "Integration did not reach end time. {} steps taken, {} GMRES iterations.",
                n_steps, n_krylov_iters
            ))
        },
        n_eval: n_f_evals,
        n_constraint_eval: n_g_evals,
        n_steps,
        n_accepted,
        n_rejected,
        n_lu: 0, // No LU decompositions with Krylov methods
        n_jac: n_jac_evals,
        method: ODEMethod::Bdf,
        dae_type: DAEType::SemiExplicit,
        index: DAEIndex::Index1,
    };

    Ok(result)
}

/// Krylov-enhanced BDF method for fully implicit DAE systems
///
/// Implements a specialized BDF method for fully implicit DAE systems that
/// uses Krylov subspace methods (GMRES) to solve the linear systems that
/// arise in the Newton iterations. This approach is particularly effective
/// for large, sparse DAE systems.
///
/// The fully implicit form is:
/// F(t, y, y') = 0
pub fn krylov_bdf_implicit_dae<F, FFunc>(
    f: FFunc,
    t_span: [F; 2],
    y0: Array1<F>,
    y_prime0: Array1<F>,
    options: DAEOptions<F>,
) -> IntegrateResult<DAEResult<F>>
where
    F: IntegrateFloat,
    FFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
{
    // Get dimensions
    let n = y0.len();

    // Storage for solution
    let mut t_values = vec![t_span[0]];
    let mut y_values = vec![y0.clone()];
    let mut y_prime_values = vec![y_prime0.clone()];

    // Current values
    let mut t_current = t_span[0];
    let mut y_current = y0.clone();
    let mut _y_prime_current = y_prime0.clone();

    // Initial step size
    let mut h = options.h0.unwrap_or_else(|| {
        let span = t_span[1] - t_span[0];
        span * F::from_f64(0.01).unwrap() // 1% of interval
    });

    // Step limits
    let min_step = options.min_step.unwrap_or_else(|| {
        let span = t_span[1] - t_span[0];
        span * F::from_f64(1e-6).unwrap() // Very small relative to interval
    });

    let max_step = options.max_step.unwrap_or_else(|| {
        let span = t_span[1] - t_span[0];
        span * F::from_f64(0.1).unwrap() // 10% of interval
    });

    // Maximum BDF order
    let max_order = options.max_order.unwrap_or(5).min(5);

    // Tolerances
    let rtol = options.rtol;
    let atol = options.atol;

    // Krylov method settings
    let gmres_tol = F::from_f64(GMRES_TOL).unwrap();
    let max_gmres_iter = options.max_newton_iterations * 3; // Allow more GMRES iterations than Newton

    // Counters for statistics
    let mut n_steps = 0;
    let mut n_accepted = 0;
    let mut n_rejected = 0;
    let mut n_f_evals = 0;
    let mut n_jac_evals = 0;
    let mut n_krylov_iters = 0;

    // BDF method coefficients for various orders
    // Alpha coefficients (exclude α_0 = 1)
    let alpha_coeffs = [
        // Order 1 (Backward Euler)
        vec![F::from_f64(-1.0).unwrap()],
        // Order 2
        vec![
            F::from_f64(-4.0 / 3.0).unwrap(),
            F::from_f64(1.0 / 3.0).unwrap(),
        ],
        // Order 3
        vec![
            F::from_f64(-18.0 / 11.0).unwrap(),
            F::from_f64(9.0 / 11.0).unwrap(),
            F::from_f64(-2.0 / 11.0).unwrap(),
        ],
        // Order 4
        vec![
            F::from_f64(-48.0 / 25.0).unwrap(),
            F::from_f64(36.0 / 25.0).unwrap(),
            F::from_f64(-16.0 / 25.0).unwrap(),
            F::from_f64(3.0 / 25.0).unwrap(),
        ],
        // Order 5
        vec![
            F::from_f64(-300.0 / 137.0).unwrap(),
            F::from_f64(300.0 / 137.0).unwrap(),
            F::from_f64(-200.0 / 137.0).unwrap(),
            F::from_f64(75.0 / 137.0).unwrap(),
            F::from_f64(-12.0 / 137.0).unwrap(),
        ],
    ];

    // Beta coefficients (for derivative approximation)
    let beta_coeffs = [
        F::from_f64(1.0).unwrap(),          // Order 1
        F::from_f64(2.0 / 3.0).unwrap(),    // Order 2
        F::from_f64(6.0 / 11.0).unwrap(),   // Order 3
        F::from_f64(12.0 / 25.0).unwrap(),  // Order 4
        F::from_f64(60.0 / 137.0).unwrap(), // Order 5
    ];

    // Current order (start with order 1 and build up)
    let mut order = 1;

    // Main integration loop
    while t_current < t_span[1] && n_steps < options.max_steps {
        // Adjust step size to hit the end point exactly if needed
        if t_current + h > t_span[1] {
            h = t_span[1] - t_current;
        }

        // New time point
        let t_new = t_current + h;

        // Build up history
        let history_start = if y_values.len() >= order {
            y_values.len() - order
        } else {
            0
        };

        let y_history = &y_values[history_start..];

        // Predict step using extrapolation
        let y_pred = predict_fully_implicit(y_history, order);

        // For the predictor's derivative, we use the BDF formula
        let y_prime_pred = if order == 1 {
            // For first-order, just use backward Euler
            (&y_pred - &y_current) / h
        } else {
            // For higher orders, compute using the BDF formula
            // y_prime = (Σ α_j * y_{n+1-j}) / (h * β)
            let mut y_prime = Array1::zeros(n);

            // Get coefficients for this order
            let alpha = &alpha_coeffs[order - 1];
            let beta = beta_coeffs[order - 1];

            // Compute the derivative approximation
            for i in 0..n {
                // First term: y_pred
                y_prime[i] = y_pred[i];

                // Historical terms: + Σ α_j * y_{n-j}
                for (j, &alpha_j) in alpha.iter().enumerate().take(order) {
                    let idx = y_values.len() - 1 - j;
                    if idx < y_values.len() {
                        y_prime[i] += alpha_j * y_values[idx][i];
                    }
                }

                // Scale by the beta coefficient
                y_prime[i] /= h * beta;
            }

            y_prime
        };

        // Store original predictions for error estimation
        let y_pred_orig = y_pred.clone();
        let _y_prime_pred_orig = y_prime_pred.clone();

        // Initialize corrector values
        let mut y_corr = y_pred;
        let mut y_prime_corr = y_prime_pred;

        // Maximum Newton iterations
        let max_newton_iter = options.max_newton_iterations;

        // Newton iteration for corrector
        let mut converged = false;
        for _iter in 0..max_newton_iter {
            // Evaluate the residual function
            let residual = f(t_new, y_corr.view(), y_prime_corr.view());
            n_f_evals += 1;

            // Compute residual norm
            let residual_norm = residual
                .iter()
                .fold(F::zero(), |acc, &val| acc + val * val)
                .sqrt();

            // Check convergence
            if residual_norm < options.newton_tol {
                converged = true;
                break;
            }

            // Compute the Jacobians of the residual function
            // We need ∂F/∂y and ∂F/∂y'
            let jac_y = compute_jacobian_y_implicit(
                &f,
                t_new,
                y_corr.view(),
                y_prime_corr.view(),
                F::from_f64(1e-8).unwrap(),
            );
            let jac_yprime = compute_jacobian_yprime_implicit(
                &f,
                t_new,
                y_corr.view(),
                y_prime_corr.view(),
                F::from_f64(1e-8).unwrap(),
            );
            n_jac_evals += 2;

            // For BDF, the derivative approximation is:
            // y' = (y - Σ α_j * y_{n-j}) / (h * β)
            // So ∂y'/∂y = 1 / (h * β)
            let beta = beta_coeffs[order - 1];
            let scale = F::one() / (h * beta);

            // Create a matrix-vector product function for GMRES
            // This computes (∂F/∂y + (∂F/∂y') * (∂y'/∂y)) * v = (∂F/∂y + (∂F/∂y') * scale) * v
            let jacobian_matvec = |v: &Array1<F>| -> Array1<F> {
                let mut result = Array1::<F>::zeros(n);

                // Compute jac_y * v
                for i in 0..n {
                    for j in 0..n {
                        result[i] += jac_y[[i, j]] * v[j];
                    }
                }

                // Add jac_yprime * scale * v
                for i in 0..n {
                    for j in 0..n {
                        result[i] += jac_yprime[[i, j]] * scale * v[j];
                    }
                }

                result
            };

            // Create a simple diagonal preconditioner
            let diag_preconditioner = |v: &Array1<F>| -> Array1<F> {
                // Extract diagonal of the combined Jacobian
                let mut diag = Array1::<F>::zeros(n);

                // For each row, compute the diagonal entry: ∂F_i/∂y_i + (∂F_i/∂y'_i) * scale
                for i in 0..n {
                    diag[i] = jac_y[[i, i]] + jac_yprime[[i, i]] * scale;
                    if diag[i].abs() < F::from_f64(1e-14).unwrap() {
                        diag[i] = F::from_f64(1e-14).unwrap() * diag[i].signum();
                    }
                    diag[i] = F::one() / diag[i];
                }

                // Apply diagonal preconditioner: P⁻¹v = D⁻¹v
                v.iter()
                    .zip(diag.iter())
                    .map(|(&vi, &di)| vi * di)
                    .collect()
            };

            // Negate the residual for the Newton step
            let neg_residual = residual.mapv(|x| -x);

            // Solve the linear system using GMRES
            let (delta_y, gmres_iterations) = match gmres_solver(
                jacobian_matvec,
                &neg_residual,
                Some(diag_preconditioner),
                gmres_tol,
                max_gmres_iter,
                GMRES_RESTART,
            ) {
                Ok((dy, iters)) => (dy, iters),
                Err(_e) => {
                    // If the linear solve fails, try with a smaller step
                    // and terminate this Newton iteration
                    h *= F::from_f64(0.5).unwrap();
                    break;
                }
            };
            n_krylov_iters += gmres_iterations;

            // Apply the Newton step with damping if needed
            let mut alpha_damp = F::one();
            let min_alpha = F::from_f64(0.1).unwrap();

            // Damped Newton iteration
            while alpha_damp >= min_alpha {
                // Apply the damped step
                let y_new = &y_corr + &(&delta_y * alpha_damp);

                // Compute the new derivative using the BDF formula
                let mut y_prime_new = Array1::zeros(n);

                // Get coefficients for this order
                let alpha = &alpha_coeffs[order - 1];
                let beta = beta_coeffs[order - 1];

                // Compute the derivative approximation
                for i in 0..n {
                    // First term: y_new
                    y_prime_new[i] = y_new[i];

                    // Historical terms: + Σ α_j * y_{n-j}
                    for j in 0..order {
                        if j < y_history.len() {
                            y_prime_new[i] += alpha[j] * y_history[y_history.len() - 1 - j][i];
                        }
                    }

                    // Scale by the beta coefficient
                    y_prime_new[i] /= h * beta;
                }

                // Evaluate the residual at the new point
                let residual_new = f(t_new, y_new.view(), y_prime_new.view());
                n_f_evals += 1;

                // Compute the new residual norm
                let residual_new_norm = residual_new
                    .iter()
                    .fold(F::zero(), |acc, &val| acc + val * val)
                    .sqrt();

                // Accept if the residual is reduced
                if residual_new_norm < residual_norm {
                    y_corr = y_new;
                    y_prime_corr = y_prime_new;
                    break;
                }

                // Reduce damping factor
                alpha_damp *= F::from_f64(0.5).unwrap();
            }

            // If damping factor got too small, the Newton iteration is not converging
            if alpha_damp < min_alpha {
                // Reduce step size and try again
                h *= F::from_f64(0.5).unwrap();
                break;
            }
        }

        // Check for convergence of the Newton iteration
        if !converged {
            // If not converged, reduce step size and try again
            h *= F::from_f64(0.5).unwrap();

            // If step size gets too small, the problem might be too stiff
            if h < min_step {
                return Err(IntegrateError::ComputationError(format!(
                    "Failed to converge at t = {}. Step size too small.",
                    t_current
                )));
            }

            n_rejected += 1;
            continue;
        }

        // Step accepted, update step count
        n_steps += 1;
        n_accepted += 1;

        // Estimate local error for step size control
        // For BDF methods, we can use the difference between the predictor and corrector
        let error = (&y_corr - &y_pred_orig).mapv(|x| x.abs());

        // Compute scaled error norm
        let mut error_norm = F::zero();
        for i in 0..n {
            let scale = atol + rtol * y_corr[i].abs();
            error_norm += (error[i] / scale).powi(2);
        }
        error_norm = (error_norm / F::from_usize(n).unwrap()).sqrt();

        // Adjust the step size based on the error estimate
        let error_order = order as i32;
        let safety = F::from_f64(0.9).unwrap(); // Safety factor

        // Calculate the optimal step size
        let h_new = if error_norm > F::zero() {
            h * safety * (F::one() / error_norm).powf(F::one() / F::from_i32(error_order).unwrap())
        } else {
            h * F::from_f64(2.0).unwrap() // Double the step size if error is 0
        };

        // Limit step size increase and decrease
        let max_increase = F::from_f64(2.0).unwrap();
        let max_decrease = F::from_f64(0.1).unwrap();

        let h_new = (h_new / h).max(max_decrease).min(max_increase) * h;
        let h_new = h_new.min(max_step).max(min_step);

        // Update step size for next iteration
        h = h_new;

        // Store the solution
        t_values.push(t_new);
        y_values.push(y_corr.clone());
        y_prime_values.push(y_prime_corr.clone());

        // Update current values
        t_current = t_new;
        y_current = y_corr;
        _y_prime_current = y_prime_corr;

        // Adjust the order based on history
        if n_steps >= 5 {
            // At this point, we have enough history to consider increasing the order
            if order < max_order {
                order = (order + 1).min(max_order);
            }
        }
    }

    // Check if we reached the end time
    let success = t_current >= t_span[1];

    // Create the empty array for algebraic variables
    // (in the fully implicit form, we don't separate differential and algebraic variables)
    let empty_array = Array1::<F>::zeros(0);
    let empty_y = vec![empty_array; t_values.len()];

    // Create result
    let result = DAEResult {
        t: t_values,
        x: y_values,
        y: empty_y,
        success,
        message: if success {
            Some(format!(
                "Successful integration. {} steps taken, {} GMRES iterations.",
                n_steps, n_krylov_iters
            ))
        } else {
            Some(format!(
                "Integration did not reach end time. {} steps taken, {} GMRES iterations.",
                n_steps, n_krylov_iters
            ))
        },
        n_eval: n_f_evals,
        n_constraint_eval: 0, // No explicit constraints in fully implicit form
        n_steps,
        n_accepted,
        n_rejected,
        n_lu: 0, // No LU decompositions with Krylov methods
        n_jac: n_jac_evals,
        method: ODEMethod::Bdf,
        dae_type: DAEType::FullyImplicit,
        index: DAEIndex::Index1,
    };

    Ok(result)
}

/// GMRES solver for solving linear systems Ax = b without explicitly forming A
///
/// # Arguments
/// * `matvec` - A function that computes the matrix-vector product Av
/// * `b` - The right-hand side vector
/// * `preconditioner` - Optional preconditioner function P⁻¹v
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
/// * `restart` - Restart parameter
///
/// # Returns
/// * The solution vector and the number of iterations taken
fn gmres_solver<F>(
    matvec: impl Fn(&Array1<F>) -> Array1<F>,
    b: &Array1<F>,
    preconditioner: Option<impl Fn(&Array1<F>) -> Array1<F>>,
    tol: F,
    max_iter: usize,
    restart: usize,
) -> IntegrateResult<(Array1<F>, usize)>
where
    F: IntegrateFloat,
{
    let n = b.len();
    let mut x = Array1::<F>::zeros(n);
    let mut total_iters = 0;

    // Apply preconditioner to b if provided
    let b_precond = match &preconditioner {
        Some(precond) => precond(b),
        None => b.clone(),
    };

    // Compute initial residual: r = P⁻¹(b - Ax)
    let r0 = if x.iter().all(|&v| v == F::zero()) {
        // If x is zero, r = P⁻¹b
        b_precond.clone()
    } else {
        // Otherwise, r = P⁻¹(b - Ax)
        let ax = matvec(&x);
        let residual = b - &ax;
        match &preconditioner {
            Some(precond) => precond(&residual),
            None => residual,
        }
    };

    let r0_norm = r0.iter().fold(F::zero(), |acc, &v| acc + v * v).sqrt();

    // Initial check for convergence or zero RHS
    if r0_norm <= tol {
        return Ok((x, 0)); // Already converged
    }

    // Scaled tolerance based on initial residual
    let scaled_tol = tol * r0_norm;

    // Main GMRES loop with restarts
    let mut cycles = 0;
    let max_cycles = max_iter.div_ceil(restart);

    while cycles < max_cycles {
        // Build the Arnoldi basis and Hessenberg matrix
        let (h, q, residual_norm, iters) =
            arnoldi_process(&matvec, &preconditioner, &r0, r0_norm, scaled_tol, restart)?;

        total_iters += iters;

        // Apply update to current solution
        // The solution update is computed by solving the least squares problem
        // min ||beta * e1 - H * y||_2 and then x += Q * y
        // We use the Givens rotations from the Arnoldi process
        let y = solve_upper_triangular(&h, &residual_norm, iters);

        // Apply the update: x += Q * y
        for i in 0..iters {
            for j in 0..n {
                x[j] += q[i][j] * y[i];
            }
        }

        // Check for convergence
        if residual_norm <= scaled_tol {
            break;
        }

        // Prepare for next iteration
        // Re-compute residual for the next restart cycle
        let ax = matvec(&x);
        let residual = b - &ax;
        let _r0 = match &preconditioner {
            Some(precond) => precond(&residual),
            None => residual,
        };

        cycles += 1;
    }

    Ok((x, total_iters))
}

/// Arnoldi process for building Krylov subspace basis
///
/// # Arguments
/// * `matvec` - Matrix-vector product function
/// * `preconditioner` - Optional preconditioner
/// * `r0` - Initial residual
/// * `r0_norm` - Norm of initial residual
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// * Upper Hessenberg matrix, Krylov vectors, residual norm, iteration count
fn arnoldi_process<F>(
    matvec: &impl Fn(&Array1<F>) -> Array1<F>,
    preconditioner: &Option<impl Fn(&Array1<F>) -> Array1<F>>,
    r0: &Array1<F>,
    r0_norm: F,
    tol: F,
    max_iter: usize,
) -> IntegrateResult<GMRESResult<F>>
where
    F: IntegrateFloat,
{
    let n = r0.len();
    let m = max_iter.min(n);

    // Allocate space for the orthonormal basis
    let mut q = Vec::with_capacity(m + 1);
    q.push(r0 / r0_norm);

    // Allocate space for the Hessenberg matrix
    let mut h = Array2::<F>::zeros((m + 1, m));

    // Store the Givens rotations
    let mut cs = Array1::<F>::zeros(m);
    let mut sn = Array1::<F>::zeros(m);

    // Initialize the right-hand side of the least squares problem
    let mut g = Array1::<F>::zeros(m + 1);
    g[0] = r0_norm;

    // Main Arnoldi iteration
    for j in 0..m {
        // Apply matrix and preconditioner: w = P⁻¹Aq_j
        let aq = matvec(&q[j]);
        let mut w = match preconditioner {
            Some(precond) => precond(&aq),
            None => aq,
        };

        // Modified Gram-Schmidt orthogonalization
        for i in 0..=j {
            h[[i, j]] = dot(&q[i], &w);
            for k in 0..n {
                w[k] -= h[[i, j]] * q[i][k];
            }
        }

        // Compute the norm of the new vector
        let w_norm = w.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();
        h[[j + 1, j]] = w_norm;

        // Check for breakdown
        if w_norm < F::from_f64(1e-14).unwrap() {
            // Early convergence or breakdown
            return Ok((h, q, g[j + 1], j + 1));
        }

        // Normalize and store the new basis vector
        q.push(w / w_norm);

        // Apply previous Givens rotations to the new column of the Hessenberg matrix
        for i in 0..j {
            let temp = h[[i, j]];
            h[[i, j]] = cs[i] * temp + sn[i] * h[[i + 1, j]];
            h[[i + 1, j]] = -sn[i] * temp + cs[i] * h[[i + 1, j]];
        }

        // Compute and apply a new Givens rotation to eliminate h[j+1][j]
        let (c, s) = givens_rotation(h[[j, j]], h[[j + 1, j]]);
        cs[j] = c;
        sn[j] = s;

        // Update the Hessenberg matrix
        let temp = h[[j, j]];
        h[[j, j]] = c * temp + s * h[[j + 1, j]];
        h[[j + 1, j]] = F::zero(); // Explicitly set to zero for numerical stability

        // Update the RHS of the least squares problem
        let temp = g[j];
        g[j] = c * temp;
        g[j + 1] = -s * temp;

        // Check for convergence
        if g[j + 1].abs() <= tol {
            return Ok((h, q, g[j + 1].abs(), j + 1));
        }
    }

    Ok((h, q, g[m].abs(), m))
}

/// Compute the dot product of two vectors
fn dot<F>(a: &Array1<F>, b: &Array1<F>) -> F
where
    F: IntegrateFloat,
{
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + ai * bi)
}

/// Compute a Givens rotation matrix that zeros out an entry
fn givens_rotation<F>(a: F, b: F) -> (F, F)
where
    F: IntegrateFloat,
{
    if b == F::zero() {
        (F::one(), F::zero())
    } else if a.abs() < b.abs() {
        let t = -a / b;
        let s = F::one() / (F::one() + t * t).sqrt();
        let c = s * t;
        (-c, s)
    } else {
        let t = -b / a;
        let c = F::one() / (F::one() + t * t).sqrt();
        let s = c * t;
        (c, s)
    }
}

/// Solve an upper triangular system Rx = b
fn solve_upper_triangular<F>(r: &Array2<F>, g: &F, n: usize) -> Array1<F>
where
    F: IntegrateFloat,
{
    let mut x = Array1::<F>::zeros(n);
    let mut g_vec = Array1::<F>::zeros(n);
    g_vec[0] = *g;

    for i in (0..n).rev() {
        let mut sum = F::zero();
        for j in (i + 1)..n {
            sum += r[[i, j]] * x[j];
        }
        x[i] = (g_vec[i] - sum) / r[[i, i]];
    }

    x
}

/// Predict the next state for semi-explicit DAE using extrapolation
fn predict_step<F>(
    x_history: &[Array1<F>],
    y_history: &[Array1<F>],
    order: usize,
    _h: F,
) -> (Array1<F>, Array1<F>)
where
    F: IntegrateFloat,
{
    let n_x = x_history[0].len();
    let n_y = y_history[0].len();

    let history_len = x_history.len();

    if history_len < 2 || order == 1 {
        // For first step or first-order method, just use constant extrapolation
        return (
            x_history[history_len - 1].clone(),
            y_history[history_len - 1].clone(),
        );
    }

    // For higher-order extrapolation, use polynomial interpolation
    let order_to_use = order.min(history_len - 1);

    // Start with the most recent point
    let mut x_pred = x_history[history_len - 1].clone();
    let mut y_pred = y_history[history_len - 1].clone();

    // Simple linear extrapolation for order 2
    if order_to_use == 1 {
        // Linear extrapolation
        for i in 0..n_x {
            x_pred[i] = x_history[history_len - 1][i]
                + (x_history[history_len - 1][i] - x_history[history_len - 2][i]);
        }

        for i in 0..n_y {
            y_pred[i] = y_history[history_len - 1][i]
                + (y_history[history_len - 1][i] - y_history[history_len - 2][i]);
        }

        return (x_pred, y_pred);
    }

    // For higher orders, use a more sophisticated extrapolation
    // Scale by order factor to improve prediction for higher orders
    let scaling = F::from_f64(1.0 + 0.3 * order_to_use as f64).unwrap();

    // Extrapolate
    for i in 0..n_x {
        x_pred[i] = x_history[history_len - 1][i]
            + (x_history[history_len - 1][i] - x_history[history_len - 2][i]) * scaling;
    }

    for i in 0..n_y {
        y_pred[i] = y_history[history_len - 1][i]
            + (y_history[history_len - 1][i] - y_history[history_len - 2][i]) * scaling;
    }

    (x_pred, y_pred)
}

/// Predict the next state for fully implicit DAE
fn predict_fully_implicit<F>(y_history: &[Array1<F>], order: usize) -> Array1<F>
where
    F: IntegrateFloat,
{
    let n = y_history[0].len();
    let history_len = y_history.len();

    if history_len < 2 || order == 1 {
        // For first step or first-order method, just use constant extrapolation
        return y_history[history_len - 1].clone();
    }

    // For higher-order extrapolation, we'll use a simple polynomial predictor
    // For simplicity, we'll just use linear extrapolation here
    // In a full implementation, higher-order predictors would be used

    let mut y_pred = y_history[history_len - 1].clone();

    // Simple linear extrapolation
    for i in 0..n {
        y_pred[i] = y_history[history_len - 1][i]
            + (y_history[history_len - 1][i] - y_history[history_len - 2][i]);
    }

    y_pred
}

/// Compute the Jacobian of a function with respect to x variables
fn compute_jacobian_x<F, Func>(
    f: &Func,
    t: F,
    x: ArrayView1<F>,
    y: ArrayView1<F>,
    epsilon: F,
) -> Array2<F>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
{
    let n_x = x.len();
    let n_f = f(t, x, y).len();
    let mut jacobian = Array2::<F>::zeros((n_f, n_x));

    // Base function value
    let f_base = f(t, x, y);

    // Compute the Jacobian using finite differences
    let mut x_perturbed = x.to_owned();

    for j in 0..n_x {
        // Compute the perturbation size based on the variable magnitude
        let h = epsilon.max(x[j].abs() * epsilon);

        // Perturb the jth component
        x_perturbed[j] = x[j] + h;

        // Evaluate the function with the perturbed variable
        let f_perturbed = f(t, x_perturbed.view(), y);

        // Reset the perturbation
        x_perturbed[j] = x[j];

        // Compute the finite difference approximation
        let col_j = (f_perturbed - &f_base) / h;

        // Store in the Jacobian
        for i in 0..n_f {
            jacobian[[i, j]] = col_j[i];
        }
    }

    jacobian
}

/// Compute the Jacobian of a function with respect to y variables
fn compute_jacobian_y<F, Func>(
    f: &Func,
    t: F,
    x: ArrayView1<F>,
    y: ArrayView1<F>,
    epsilon: F,
) -> Array2<F>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
{
    let n_y = y.len();
    let n_f = f(t, x, y).len();
    let mut jacobian = Array2::<F>::zeros((n_f, n_y));

    // Base function value
    let f_base = f(t, x, y);

    // Compute the Jacobian using finite differences
    let mut y_perturbed = y.to_owned();

    for j in 0..n_y {
        // Compute the perturbation size based on the variable magnitude
        let h = epsilon.max(y[j].abs() * epsilon);

        // Perturb the jth component
        y_perturbed[j] = y[j] + h;

        // Evaluate the function with the perturbed variable
        let f_perturbed = f(t, x, y_perturbed.view());

        // Reset the perturbation
        y_perturbed[j] = y[j];

        // Compute the finite difference approximation
        let col_j = (f_perturbed - &f_base) / h;

        // Store in the Jacobian
        for i in 0..n_f {
            jacobian[[i, j]] = col_j[i];
        }
    }

    jacobian
}

/// Compute the Jacobian of a function with respect to y for implicit DAE
fn compute_jacobian_y_implicit<F, Func>(
    f: &Func,
    t: F,
    y: ArrayView1<F>,
    y_prime: ArrayView1<F>,
    epsilon: F,
) -> Array2<F>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
{
    let n = y.len();
    let n_f = f(t, y, y_prime).len();
    let mut jacobian = Array2::<F>::zeros((n_f, n));

    // Base function value
    let f_base = f(t, y, y_prime);

    // Compute the Jacobian using finite differences
    let mut y_perturbed = y.to_owned();

    for j in 0..n {
        // Compute the perturbation size based on the variable magnitude
        let h = epsilon.max(y[j].abs() * epsilon);

        // Perturb the jth component
        y_perturbed[j] = y[j] + h;

        // Evaluate the function with the perturbed variable
        let f_perturbed = f(t, y_perturbed.view(), y_prime);

        // Reset the perturbation
        y_perturbed[j] = y[j];

        // Compute the finite difference approximation
        let col_j = (f_perturbed - &f_base) / h;

        // Store in the Jacobian
        for i in 0..n_f {
            jacobian[[i, j]] = col_j[i];
        }
    }

    jacobian
}

/// Compute the Jacobian of a function with respect to y' for implicit DAE
fn compute_jacobian_yprime_implicit<F, Func>(
    f: &Func,
    t: F,
    y: ArrayView1<F>,
    y_prime: ArrayView1<F>,
    epsilon: F,
) -> Array2<F>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F>,
{
    let n = y_prime.len();
    let n_f = f(t, y, y_prime).len();
    let mut jacobian = Array2::<F>::zeros((n_f, n));

    // Base function value
    let f_base = f(t, y, y_prime);

    // Compute the Jacobian using finite differences
    let mut y_prime_perturbed = y_prime.to_owned();

    for j in 0..n {
        // Compute the perturbation size based on the variable magnitude
        let h = epsilon.max(y_prime[j].abs() * epsilon);

        // Perturb the jth component
        y_prime_perturbed[j] = y_prime[j] + h;

        // Evaluate the function with the perturbed variable
        let f_perturbed = f(t, y, y_prime_perturbed.view());

        // Reset the perturbation
        y_prime_perturbed[j] = y_prime[j];

        // Compute the finite difference approximation
        let col_j = (f_perturbed - &f_base) / h;

        // Store in the Jacobian
        for i in 0..n_f {
            jacobian[[i, j]] = col_j[i];
        }
    }

    jacobian
}
