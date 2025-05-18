//! Implicit ODE solver methods
//!
//! This module implements implicit methods for solving ODEs,
//! including the Backward Differentiation Formula (BDF) method
//! and the Radau IIA method.

use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::types::{ODEMethod, ODEOptions, ODEResult};
use crate::IntegrateFloat;
use ndarray::{Array1, Array2, ArrayView1};

/// Solve ODE using the Backward Differentiation Formula (BDF) method
///
/// BDF is an implicit multistep method particularly suited for stiff problems.
/// It is more computationally expensive per step than explicit methods but
/// can take much larger steps for stiff problems, resulting in overall better
/// performance for such systems.
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
pub fn bdf_method<F, Func>(
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

    // Only use coefficients for the requested order
    let coeffs = &bdf_coefs[order - 1];

    // Main integration loop
    while t < t_end && step_count < opts.max_steps {
        // Adjust step size for the last step if needed
        if t + h > t_end {
            h = t_end - t;
        }

        // Limit step size to bounds
        h = h.min(max_step).max(min_step);

        // Create the system matrix for the implicit equation
        // For BDF, we need to solve the nonlinear system:
        // c_0 * y_{n+1} - h * f(t_{n+1}, y_{n+1}) - sum_{j=1}^p c_j * y_{n+1-j} = 0

        // Initialize the residual function
        let next_t = t + h;

        // Predict initial value using explicit extrapolation
        // This provides a better starting guess for Newton iteration
        let mut y_pred = y.clone();

        // If we have multiple previous points, use them for prediction
        if order > 1 && y_values.len() >= order {
            // Simple linear extrapolation for BDF2
            if order == 2 {
                let y_nm1 = &y_values[y_values.len() - 2];
                y_pred = y.clone() * F::from_f64(2.0).unwrap() - y_nm1 * F::one();
            } else {
                // For higher orders, use a quadratic or higher extrapolation
                // This is approximate but sufficient for an initial guess
                let y_curr = &y_values[y_values.len() - 1];
                let y_prev = &y_values[y_values.len() - 2];
                let y_prev2 = &y_values[y_values.len() - 3];

                let a = F::one();
                let b = F::from_f64(2.0).unwrap();

                y_pred = y_curr * (a + b) - y_prev * b + y_prev2 * (F::one() - a);
            }
        }

        // Newton's method for solving the nonlinear system
        let mut y_next = y_pred.clone();
        let mut converged = false;
        let mut iter_count = 0;
        let max_newton_iters = 10; // Maximum iterations for Newton's method

        while iter_count < max_newton_iters {
            // Evaluate function at the current iterate
            let f_eval = f(next_t, y_next.view());
            func_evals += 1;

            // Compute residual: c_0 * y_{n+1} - h * f(t_{n+1}, y_{n+1}) - sum_{j=1}^p c_j * y_{n+1-j}
            let mut residual = y_next.clone() * coeffs[0];

            // Previous values contribution
            for (j, coeff) in coeffs.iter().enumerate().skip(1).take(order) {
                if j <= y_values.len() {
                    let idx = y_values.len() - j;
                    residual = residual - y_values[idx].clone() * *coeff;
                }
            }

            // Subtract h * f(t_{n+1}, y_{n+1})
            residual = residual - f_eval.clone() * h;

            // Compute Newton step
            // In a full implementation, we would compute the Jacobian:
            // J = c_0 * I - h * df/dy
            // However, computing the actual Jacobian is complex
            // For simplicity, we'll use a finite difference approximation

            // Create approximate Jacobian using finite differences
            let eps = F::from_f64(1e-8).unwrap();
            let mut jacobian = Array2::<F>::zeros((n_dim, n_dim));
            n_jac += 1;

            for i in 0..n_dim {
                let mut y_perturbed = y_next.clone();
                y_perturbed[i] += eps;

                let f_perturbed = f(next_t, y_perturbed.view());
                func_evals += 1;

                for j in 0..n_dim {
                    // Finite difference approximation of df_j/dy_i
                    let df_dy = (f_perturbed[j] - f_eval[j]) / eps;

                    // J_{ji} = c_0 * δ_{ji} - h * df_j/dy_i
                    jacobian[[j, i]] = if i == j {
                        coeffs[0] - h * df_dy
                    } else {
                        -h * df_dy
                    };
                }
            }

            // For small systems (n_dim typically <= 10), we can use a simple approach
            // instead of full matrix solver to avoid dependency issues

            // For a 1D system, we can directly solve without any matrix inversion
            if n_dim == 1 {
                // For scalar case, J is just a number, and delta_y = -residual / J
                if jacobian[[0, 0]].abs() < F::from_f64(1e-10).unwrap() {
                    // Nearly singular, reduce step size and try again
                    h *= F::from_f64(0.5).unwrap();
                    if h < min_step {
                        return Err(IntegrateError::ConvergenceError(
                            "Newton iteration failed to converge with minimum step size"
                                .to_string(),
                        ));
                    }
                    iter_count = 0;
                    continue;
                }

                // Direct solution for scalar case
                let delta_y0 = residual[0] / jacobian[[0, 0]];
                y_next[0] -= delta_y0;
            }
            // For larger systems, use Gaussian elimination
            else {
                // Implement Gaussian elimination for larger systems
                // Copy the matrix and right-hand side for manipulation
                let mut aug = Array2::<F>::zeros((n_dim, n_dim + 1));
                for i in 0..n_dim {
                    for j in 0..n_dim {
                        aug[[i, j]] = jacobian[[i, j]];
                    }
                    aug[[i, n_dim]] = residual[i];
                }

                // Gaussian elimination with partial pivoting
                n_lu += 1;
                for i in 0..n_dim {
                    // Find pivot
                    let mut max_idx = i;
                    let mut max_val = aug[[i, i]].abs();

                    for j in i + 1..n_dim {
                        if aug[[j, i]].abs() > max_val {
                            max_idx = j;
                            max_val = aug[[j, i]].abs();
                        }
                    }

                    // Check if the matrix is singular
                    if max_val < F::from_f64(1e-10).unwrap() {
                        // Nearly singular matrix, reduce step size and try again
                        h *= F::from_f64(0.5).unwrap();
                        if h < min_step {
                            return Err(IntegrateError::ConvergenceError(
                                "Newton iteration failed to converge with minimum step size"
                                    .to_string(),
                            ));
                        }
                        iter_count = 0;
                        continue;
                    }

                    // Swap rows if necessary
                    if max_idx != i {
                        for j in 0..n_dim + 1 {
                            let temp = aug[[i, j]];
                            aug[[i, j]] = aug[[max_idx, j]];
                            aug[[max_idx, j]] = temp;
                        }
                    }

                    // Eliminate below
                    for j in i + 1..n_dim {
                        let factor = aug[[j, i]] / aug[[i, i]];
                        for k in i..n_dim + 1 {
                            aug[[j, k]] = aug[[j, k]] - factor * aug[[i, k]];
                        }
                    }
                }

                // Back substitution
                let mut delta_y = Array1::<F>::zeros(n_dim);
                for i in (0..n_dim).rev() {
                    let mut sum = aug[[i, n_dim]];
                    for j in i + 1..n_dim {
                        sum -= aug[[i, j]] * delta_y[j];
                    }
                    delta_y[i] = sum / aug[[i, i]];
                }

                // Update solution
                for i in 0..n_dim {
                    y_next[i] -= delta_y[i];
                }
            }

            // Check convergence
            let newton_tol = F::from_f64(1e-8).unwrap();
            let mut err = F::zero();
            for i in 0..n_dim {
                let sc = opts.atol + opts.rtol * y_next[i].abs();
                let e: F = residual[i] / sc;
                err = err.max(e.abs());
            }

            if err <= newton_tol {
                converged = true;
                break;
            }

            iter_count += 1;
        }

        newton_iters += F::from(iter_count).unwrap();

        if converged {
            // Step accepted
            t = next_t;
            y = y_next;

            // Store results
            t_values.push(t);
            y_values.push(y.clone());

            step_count += 1;
            accepted_steps += 1;

            // Adjust step size based on Newton convergence
            // If we converged quickly, increase step size
            if iter_count <= 2 {
                h *= F::from_f64(1.2).unwrap().min(F::from_f64(5.0).unwrap());
            }
            // If we needed many iterations, decrease step size
            else if iter_count >= max_newton_iters - 1 {
                h *= F::from_f64(0.8).unwrap().max(F::from_f64(0.2).unwrap());
            }
        } else {
            // Newton iteration failed to converge, reduce step size
            h *= F::from_f64(0.5).unwrap();
            if h < min_step {
                return Err(IntegrateError::ConvergenceError(
                    "Newton iteration failed to converge with minimum step size".to_string(),
                ));
            }
            rejected_steps += 1;
        }
    }

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
        n_accepted: accepted_steps,
        n_rejected: rejected_steps,
        n_lu,
        n_jac,
        method: ODEMethod::Bdf,
    })
}

/// Solve ODE using the Radau IIA method
///
/// Radau IIA is an implicit Runge-Kutta method with high stability properties,
/// making it suitable for stiff problems. It is an L-stable method with high
/// order of accuracy.
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
pub fn radau_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
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

    // Radau IIA 3-stage method (5th order)
    // Butcher tableau for Radau IIA (3 stages)
    // c_i | a_ij
    // ----------
    //     | b_j
    //
    // c = [4-sqrt(6))/10, (4+sqrt(6))/10, 1]
    // Exact values would be irrational, so we use high precision approximations

    let c1 = F::from_f64(0.1550510257).unwrap();
    let c2 = F::from_f64(0.6449489743).unwrap();
    let c3 = F::one();

    // Runge-Kutta matrix A (coefficients a_ij)
    // We're using a 3-stage Radau IIA method
    let a11 = F::from_f64(0.1968154772).unwrap();
    let a12 = F::from_f64(-0.0678338608).unwrap();
    let a13 = F::from_f64(-0.0207959730).unwrap();

    let a21 = F::from_f64(0.3944243147).unwrap();
    let a22 = F::from_f64(0.2921005631).unwrap();
    let a23 = F::from_f64(0.0416635118).unwrap();

    let a31 = F::from_f64(0.3764030627).unwrap();
    let a32 = F::from_f64(0.5124858261).unwrap();
    let a33 = F::from_f64(0.1111111111).unwrap();

    // Weight coefficients b_j (same as last row of A for Radau IIA)
    let b1 = a31;
    let b2 = a32;
    let b3 = a33;

    // Integration variables
    let mut t = t_start;
    let mut y = y0.clone();
    let mut h = h0;

    // Result storage
    let mut t_values = vec![t];
    let mut y_values = vec![y.clone()];

    // Statistics
    let mut func_evals = 0;
    let mut step_count = 0;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;
    let mut n_lu = 0;
    let mut n_jac = 0;

    // Newton iteration parameters
    let newton_tol = F::from_f64(1e-8).unwrap();
    let max_newton_iters = 10;

    // Main integration loop
    while t < t_end && step_count < opts.max_steps {
        // Adjust step size for the last step if needed
        if t + h > t_end {
            h = t_end - t;
        }

        // Limit step size to bounds
        h = h.min(max_step).max(min_step);

        // Stage values
        let t1 = t + c1 * h;
        let t2 = t + c2 * h;
        let t3 = t + c3 * h;

        // Initial guess for stages (simple extrapolation from current state)
        let mut k1 = y.clone();
        let mut k2 = y.clone();
        let mut k3 = y.clone();

        // Newton's method to solve for stage values
        let mut converged = false;
        let mut iter_count = 0;

        // Newton iteration loop
        while iter_count < max_newton_iters {
            // Evaluate function at each stage
            let f1 = f(t1, k1.view());
            let f2 = f(t2, k2.view());
            let f3 = f(t3, k3.view());
            func_evals += 3;

            // Compute residuals for each stage
            // r_i = k_i - y_n - h * sum_j(a_ij * f_j)
            let mut r1 = k1.clone();
            r1 -= &y;
            r1 = r1 - (&f1 * (h * a11) + &f2 * (h * a12) + &f3 * (h * a13));

            let mut r2 = k2.clone();
            r2 -= &y;
            r2 = r2 - (&f1 * (h * a21) + &f2 * (h * a22) + &f3 * (h * a23));

            let mut r3 = k3.clone();
            r3 -= &y;
            r3 = r3 - (&f1 * (h * a31) + &f2 * (h * a32) + &f3 * (h * a33));

            // Check for convergence
            let mut max_res = F::zero();
            for i in 0..n_dim {
                let sc = opts.atol + opts.rtol * y[i].abs();
                max_res = max_res.max(r1[i].abs() / sc);
                max_res = max_res.max(r2[i].abs() / sc);
                max_res = max_res.max(r3[i].abs() / sc);
            }

            if max_res <= newton_tol {
                converged = true;
                break;
            }

            // Construct Jacobian for Newton iteration
            // For simplicity, we'll use a block-diagonal approximation
            // Each block corresponds to a single component across all stages
            n_jac += 1;

            // For small system sizes, we can use a direct approach for each component
            for i in 0..n_dim {
                // Extract i-th component for all stages
                let mut yi = Array1::<F>::zeros(3);
                let mut ri = Array1::<F>::zeros(3);
                yi[0] = k1[i];
                yi[1] = k2[i];
                yi[2] = k3[i];
                ri[0] = r1[i];
                ri[1] = r2[i];
                ri[2] = r3[i];

                // Approximate Jacobian for this component using finite differences
                let eps = F::from_f64(1e-8).unwrap();
                let mut jac = Array2::<F>::zeros((3, 3));

                // Disturb each stage value for this component
                for j in 0..3 {
                    let mut k1_perturbed = k1.clone();
                    let mut k2_perturbed = k2.clone();
                    let mut k3_perturbed = k3.clone();

                    if j == 0 {
                        k1_perturbed[i] += eps;
                    } else if j == 1 {
                        k2_perturbed[i] += eps;
                    } else {
                        k3_perturbed[i] += eps;
                    }

                    // Evaluate function with perturbed values
                    let f1_perturbed = f(t1, k1_perturbed.view());
                    let f2_perturbed = f(t2, k2_perturbed.view());
                    let f3_perturbed = f(t3, k3_perturbed.view());
                    func_evals += 3;

                    // Compute perturbed residuals
                    let mut r1_perturbed = k1_perturbed.clone();
                    r1_perturbed -= &y;
                    r1_perturbed = r1_perturbed
                        - (&f1_perturbed * (h * a11)
                            + &f2_perturbed * (h * a12)
                            + &f3_perturbed * (h * a13));

                    let mut r2_perturbed = k2_perturbed.clone();
                    r2_perturbed -= &y;
                    r2_perturbed = r2_perturbed
                        - (&f1_perturbed * (h * a21)
                            + &f2_perturbed * (h * a22)
                            + &f3_perturbed * (h * a23));

                    let mut r3_perturbed = k3_perturbed.clone();
                    r3_perturbed -= &y;
                    r3_perturbed = r3_perturbed
                        - (&f1_perturbed * (h * a31)
                            + &f2_perturbed * (h * a32)
                            + &f3_perturbed * (h * a33));

                    // Finite difference approximation of Jacobian
                    jac[[0, j]] = (r1_perturbed[i] - r1[i]) / eps;
                    jac[[1, j]] = (r2_perturbed[i] - r2[i]) / eps;
                    jac[[2, j]] = (r3_perturbed[i] - r3[i]) / eps;
                }

                // Solve 3x3 system for this component
                n_lu += 1;

                // Augmented matrix for Gaussian elimination
                let mut aug = Array2::<F>::zeros((3, 4));
                for j in 0..3 {
                    for k in 0..3 {
                        aug[[j, k]] = jac[[j, k]];
                    }
                    aug[[j, 3]] = ri[j];
                }

                // Gaussian elimination with partial pivoting (for 3x3 system)
                for j in 0..3 {
                    // Find pivot
                    let mut max_idx = j;
                    let mut max_val = aug[[j, j]].abs();

                    for k in j + 1..3 {
                        if aug[[k, j]].abs() > max_val {
                            max_idx = k;
                            max_val = aug[[k, j]].abs();
                        }
                    }

                    // Check for singularity
                    if max_val < F::from_f64(1e-10).unwrap() {
                        // Nearly singular matrix, reduce step size and try again
                        h *= F::from_f64(0.5).unwrap();
                        if h < min_step {
                            return Err(IntegrateError::ConvergenceError(
                                "Newton iteration failed to converge with minimum step size"
                                    .to_string(),
                            ));
                        }
                        iter_count = 0;
                        continue;
                    }

                    // Swap rows if necessary
                    if max_idx != j {
                        for k in 0..4 {
                            let temp = aug[[j, k]];
                            aug[[j, k]] = aug[[max_idx, k]];
                            aug[[max_idx, k]] = temp;
                        }
                    }

                    // Eliminate below
                    for k in j + 1..3 {
                        let factor = aug[[k, j]] / aug[[j, j]];
                        for l in j..4 {
                            aug[[k, l]] = aug[[k, l]] - factor * aug[[j, l]];
                        }
                    }
                }

                // Back substitution
                let mut delta = Array1::<F>::zeros(3);
                for j in (0..3).rev() {
                    let mut sum = aug[[j, 3]];
                    for k in j + 1..3 {
                        sum -= aug[[j, k]] * delta[k];
                    }
                    delta[j] = sum / aug[[j, j]];
                }

                // Update stage values
                k1[i] -= delta[0];
                k2[i] -= delta[1];
                k3[i] -= delta[2];
            }

            iter_count += 1;
        }

        if converged {
            // Step accepted
            // Compute the next state using the Butcher tableau weights
            let f1 = f(t1, k1.view());
            let f2 = f(t2, k2.view());
            let f3 = f(t3, k3.view());
            func_evals += 3;

            let y_next = &y + &(&f1 * (h * b1) + &f2 * (h * b2) + &f3 * (h * b3));

            // Update state
            t += h;
            y = y_next;

            // Store results
            t_values.push(t);
            y_values.push(y.clone());

            step_count += 1;
            accepted_steps += 1;

            // Adjust step size based on convergence
            if iter_count <= 2 {
                // Fast convergence, increase step size
                h *= F::from_f64(1.2).unwrap().min(F::from_f64(5.0).unwrap());
            } else if iter_count >= max_newton_iters - 1 {
                // Slow convergence, decrease step size
                h *= F::from_f64(0.8).unwrap().max(F::from_f64(0.2).unwrap());
            }
        } else {
            // Step rejected, reduce step size
            h *= F::from_f64(0.5).unwrap();
            if h < min_step {
                return Err(IntegrateError::ConvergenceError(
                    "Newton iteration failed to converge with minimum step size".to_string(),
                ));
            }
            rejected_steps += 1;
        }
    }

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
        n_accepted: accepted_steps,
        n_rejected: rejected_steps,
        n_lu,
        n_jac,
        method: ODEMethod::Radau,
    })
}
