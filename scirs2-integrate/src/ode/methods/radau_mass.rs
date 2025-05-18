//! Radau method with mass matrix support
//!
//! This module implements the Radau IIA method for solving ODEs
//! with support for mass matrices of the form M(t,y)·y' = f(t,y).

use crate::error::IntegrateResult;
use crate::ode::types::{MassMatrix, MassMatrixType, ODEMethod, ODEOptions, ODEResult};
use crate::ode::utils::common::calculate_error_weights;
use crate::ode::utils::dense_output::DenseSolution;
use crate::ode::utils::interpolation::ContinuousOutputMethod;
use crate::ode::utils::jacobian;
use crate::ode::utils::linear_solvers::solve_linear_system;
use crate::ode::utils::mass_matrix;
use crate::IntegrateFloat;
use ndarray::{Array1, Array2, ArrayView1};

/// Solve an ODE with mass matrix using the Radau IIA method
///
/// Radau IIA is an implicit Runge-Kutta method of order 5
/// with a 3-stage implementation. It is A-stable and L-stable,
/// making it well-suited for stiff problems.
///
/// This version supports mass matrices of the form M(t,y)·y' = f(t,y).
///
/// # Arguments
///
/// * `f` - ODE function: f(t, y) where M·y' = f(t,y)
/// * `t_span` - Time span [t_start, t_end]
/// * `y0` - Initial condition
/// * `mass_matrix` - Mass matrix specification
/// * `opts` - Solver options
///
/// # Returns
///
/// The solution as an ODEResult or an error
pub fn radau_method_with_mass<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    mass_matrix: MassMatrix<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + std::iter::Sum,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Initialize
    let [t_start, t_end] = t_span;
    let n_dim = y0.len();

    // Verify mass matrix compatibility
    mass_matrix::check_mass_compatibility(&mass_matrix, t_start, y0.view())?;

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
    let mut dy_values = Vec::new(); // Store derivatives for dense output

    // Compute initial derivative for dense output if enabled
    if opts.dense_output {
        // For the initial point, we compute f(t_0, y_0) / M
        let f_y0 = f(t, y.view());
        let dy0 = mass_matrix::solve_mass_system(&mass_matrix, t, y.view(), f_y0.view())?;
        dy_values.push(dy0);
    }

    // Statistics
    let mut func_evals = 1; // Counted the initial derivative
    let mut step_count = 0;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;
    let mut n_lu = 0;
    let mut n_jac = 0;

    // Error control
    let rtol = opts.rtol;
    let atol = opts.atol;

    // Newton iteration parameters
    let newton_tol = F::from_f64(1e-8).unwrap();
    let max_newton_iters = 10;

    // Create a Jacobian approximation for Newton iteration
    let mut jac_option = None;

    // Main integration loop
    while t < t_end && step_count < opts.max_steps {
        // Adjust step size for the last step if needed
        if t + h > t_end {
            h = t_end - t;
        }

        // Limit step size to bounds
        h = h.min(max_step).max(min_step);

        // Stage time points for this step
        let t1 = t + c1 * h;
        let t2 = t + c2 * h;
        let t3 = t + c3 * h; // This is t + h

        // Step counter
        step_count += 1;

        // Calculate the current f(t, y) as a starting point
        let f_current = f(t, y.view());
        func_evals += 1;

        // Initial guess for stage values using explicit Euler
        // (using mass matrix inverse if needed)
        let dy = mass_matrix::solve_mass_system(&mass_matrix, t, y.view(), f_current.view())?;
        func_evals += 1;

        let mut k1 = y.clone() + dy.clone() * (h * c1);
        let mut k2 = y.clone() + dy.clone() * (h * c2);
        let mut k3 = y.clone() + dy.clone() * h;

        // Weights for error estimation
        let error_weights = calculate_error_weights(&y, atol, rtol);

        // Compute Jacobian for Newton iteration
        // For mass matrix problems, we need both df/dy and dM/dy if state-dependent
        let mut compute_new_jacobian = true;
        let mut newton_converged = false;
        let mut newton_iterations = 0;

        // For mass matrices, we have a slightly different system
        // We're solving: M·(k_i - y)/h - sum(a_ij·f(t_j, k_j)) = 0
        // We need the Jacobian of this system for Newton's method

        // Newton iteration loop
        while !newton_converged && newton_iterations < max_newton_iters {
            newton_iterations += 1;

            // Compute the current F values at each stage
            let f1 = f(t1, k1.view());
            let f2 = f(t2, k2.view());
            let f3 = f(t3, k3.view());
            func_evals += 3;

            // Compute the residuals at each stage
            // r_i = M·(k_i - y)/h - sum(a_ij·f(t_j, k_j))

            // First, get the mass matrix at each stage
            let m1 = mass_matrix.evaluate(t1, k1.view());
            let m2 = mass_matrix.evaluate(t2, k2.view());
            let m3 = mass_matrix.evaluate(t3, k3.view());

            // For identity mass matrix, we can simplify
            if mass_matrix.matrix_type == MassMatrixType::Identity {
                // Simplified residual for identity mass matrix
                let r1 = (k1.clone() - y.clone()) / h
                    - (f1.clone() * a11 + f2.clone() * a12 + f3.clone() * a13);
                let r2 = (k2.clone() - y.clone()) / h
                    - (f1.clone() * a21 + f2.clone() * a22 + f3.clone() * a23);
                let r3 = (k3.clone() - y.clone()) / h
                    - (f1.clone() * a31 + f2.clone() * a32 + f3.clone() * a33);

                // Check convergence
                let error_norm = (r1
                    .iter()
                    .zip(error_weights.iter())
                    .map(|(r, &w)| (*r / w).powi(2))
                    .sum::<F>()
                    + r2.iter()
                        .zip(error_weights.iter())
                        .map(|(r, &w)| (*r / w).powi(2))
                        .sum::<F>()
                    + r3.iter()
                        .zip(error_weights.iter())
                        .map(|(r, &w)| (*r / w).powi(2))
                        .sum::<F>())
                .sqrt()
                    / F::from_f64(3.0).unwrap().sqrt();

                if error_norm < newton_tol {
                    newton_converged = true;
                    break;
                }

                // Compute Jacobian if needed
                if compute_new_jacobian {
                    let jacobian_matrix = jacobian::finite_difference_jacobian(
                        &f,
                        t3,
                        &k3,
                        &f3,
                        F::from_f64(1e-8).unwrap(),
                    );
                    jac_option = Some(jacobian_matrix);
                    compute_new_jacobian = false;
                    n_jac += 1;
                }

                // Get Jacobian
                let jac = jac_option.as_ref().unwrap();

                // Construct the system Jacobian for Newton iteration
                // J_i = I/h - a_ii·J
                // Where J is the Jacobian of f with respect to y
                let mut j1 = Array2::<F>::eye(n_dim);
                let mut j2 = Array2::<F>::eye(n_dim);
                let mut j3 = Array2::<F>::eye(n_dim);

                for i in 0..n_dim {
                    for j in 0..n_dim {
                        j1[[i, j]] = if i == j { F::one() / h } else { F::zero() };
                        j1[[i, j]] -= a11 * jac[[i, j]];

                        j2[[i, j]] = if i == j { F::one() / h } else { F::zero() };
                        j2[[i, j]] -= a22 * jac[[i, j]];

                        j3[[i, j]] = if i == j { F::one() / h } else { F::zero() };
                        j3[[i, j]] -= a33 * jac[[i, j]];
                    }
                }

                // Solve the linear systems to get Newton updates
                let dk1 = solve_linear_system(&j1.view(), &r1.view())?;
                let dk2 = solve_linear_system(&j2.view(), &r2.view())?;
                let dk3 = solve_linear_system(&j3.view(), &r3.view())?;
                n_lu += 3;

                // Update the stage values
                k1 -= &dk1;
                k2 -= &dk2;
                k3 -= &dk3;
            } else {
                // For non-identity mass matrices, we need to evaluate M·(k_i - y)/h
                let r1; // = Array1::<F>::zeros(n_dim);
                let r2; // = Array1::<F>::zeros(n_dim);
                let r3; // = Array1::<F>::zeros(n_dim);

                if let Some(ref m1_matrix) = m1 {
                    let diff1 = (&k1 - &y) / h;
                    r1 = m1_matrix.dot(&diff1)
                        - (f1.clone() * a11 + f2.clone() * a12 + f3.clone() * a13);
                } else {
                    // Identity mass matrix
                    r1 = (k1.clone() - y.clone()) / h
                        - (f1.clone() * a11 + f2.clone() * a12 + f3.clone() * a13);
                }

                if let Some(ref m2_matrix) = m2 {
                    let diff2 = (&k2 - &y) / h;
                    r2 = m2_matrix.dot(&diff2)
                        - (f1.clone() * a21 + f2.clone() * a22 + f3.clone() * a23);
                } else {
                    // Identity mass matrix
                    r2 = (k2.clone() - y.clone()) / h
                        - (f1.clone() * a21 + f2.clone() * a22 + f3.clone() * a23);
                }

                if let Some(ref m3_matrix) = m3 {
                    let diff3 = (&k3 - &y) / h;
                    r3 = m3_matrix.dot(&diff3)
                        - (f1.clone() * a31 + f2.clone() * a32 + f3.clone() * a33);
                } else {
                    // Identity mass matrix
                    r3 = (k3.clone() - y.clone()) / h
                        - (f1.clone() * a31 + f2.clone() * a32 + f3.clone() * a33);
                }

                // Check convergence
                let error_norm = (r1
                    .iter()
                    .zip(error_weights.iter())
                    .map(|(r, &w)| (*r / w).powi(2))
                    .sum::<F>()
                    + r2.iter()
                        .zip(error_weights.iter())
                        .map(|(r, &w)| (*r / w).powi(2))
                        .sum::<F>()
                    + r3.iter()
                        .zip(error_weights.iter())
                        .map(|(r, &w)| (*r / w).powi(2))
                        .sum::<F>())
                .sqrt()
                    / F::from_f64(3.0).unwrap().sqrt();

                if error_norm < newton_tol {
                    newton_converged = true;
                    break;
                }

                // For non-identity mass matrices, we need a more complex Newton system
                // The Jacobian is J_i = M/h - a_ii·df/dy - (k_i - y)/h·dM/dy

                // Compute Jacobian of f if needed
                if compute_new_jacobian {
                    let jacobian_matrix = jacobian::finite_difference_jacobian(
                        &f,
                        t3,
                        &k3,
                        &f3,
                        F::from_f64(1e-8).unwrap(),
                    );
                    jac_option = Some(jacobian_matrix);
                    compute_new_jacobian = false;
                    n_jac += 1;
                }

                // Get Jacobian
                let jac = jac_option.as_ref().unwrap();

                // This is a simplified approach that works for constant and time-dependent mass matrices
                // For state-dependent mass matrices, a more complex approach is needed
                // that includes dM/dy

                // Construct Newton iteration matrices
                let mut j1 = Array2::<F>::zeros((n_dim, n_dim));
                let mut j2 = Array2::<F>::zeros((n_dim, n_dim));
                let mut j3 = Array2::<F>::zeros((n_dim, n_dim));

                if let Some(ref m1_matrix) = m1 {
                    for i in 0..n_dim {
                        for j in 0..n_dim {
                            j1[[i, j]] = m1_matrix[[i, j]] / h - a11 * jac[[i, j]];
                        }
                    }
                } else {
                    // Identity mass matrix
                    for i in 0..n_dim {
                        for j in 0..n_dim {
                            j1[[i, j]] = if i == j { F::one() / h } else { F::zero() };
                            j1[[i, j]] -= a11 * jac[[i, j]];
                        }
                    }
                }

                if let Some(ref m2_matrix) = m2 {
                    for i in 0..n_dim {
                        for j in 0..n_dim {
                            j2[[i, j]] = m2_matrix[[i, j]] / h - a22 * jac[[i, j]];
                        }
                    }
                } else {
                    // Identity mass matrix
                    for i in 0..n_dim {
                        for j in 0..n_dim {
                            j2[[i, j]] = if i == j { F::one() / h } else { F::zero() };
                            j2[[i, j]] -= a22 * jac[[i, j]];
                        }
                    }
                }

                if let Some(ref m3_matrix) = m3 {
                    for i in 0..n_dim {
                        for j in 0..n_dim {
                            j3[[i, j]] = m3_matrix[[i, j]] / h - a33 * jac[[i, j]];
                        }
                    }
                } else {
                    // Identity mass matrix
                    for i in 0..n_dim {
                        for j in 0..n_dim {
                            j3[[i, j]] = if i == j { F::one() / h } else { F::zero() };
                            j3[[i, j]] -= a33 * jac[[i, j]];
                        }
                    }
                }

                // Solve the linear systems to get Newton updates
                let dk1 = solve_linear_system(&j1.view(), &r1.view())?;
                let dk2 = solve_linear_system(&j2.view(), &r2.view())?;
                let dk3 = solve_linear_system(&j3.view(), &r3.view())?;
                n_lu += 3;

                // Update the stage values
                k1 -= &dk1;
                k2 -= &dk2;
                k3 -= &dk3;
            }
        }

        // Check if Newton iteration converged
        if !newton_converged {
            // Reduce step size and try again
            h *= F::from_f64(0.5).unwrap();
            rejected_steps += 1;
            continue;
        }

        // Compute new solution
        let y_new =
            y.clone() + (f(t1, k1.view()) * b1 + f(t2, k2.view()) * b2 + f(t3, k3.view()) * b3) * h;
        func_evals += 3;

        // Estimate error using embedded method
        // For Radau IIA, we can use the difference between the last stage and the solution
        let error = &k3 - &y_new;

        // Compute error norm
        let error_norm = error
            .iter()
            .zip(error_weights.iter())
            .map(|(e, &w)| (*e / w).powi(2))
            .sum::<F>()
            .sqrt();

        // Determine if step is acceptable
        if error_norm <= F::one() {
            // Accept the step
            t += h;
            y = y_new;

            // Store the result
            t_values.push(t);
            y_values.push(y.clone());

            // For dense output, store the derivative
            if opts.dense_output {
                let f_y = f(t, y.view());
                let dy = mass_matrix::solve_mass_system(&mass_matrix, t, y.view(), f_y.view())?;
                dy_values.push(dy);
                func_evals += 1;
            }

            accepted_steps += 1;

            // Increase step size for next step if error is small
            if error_norm < F::from_f64(0.1).unwrap() {
                h *= F::from_f64(2.0).unwrap();
            }
        } else {
            // Reject the step and reduce step size
            let factor = F::from_f64(0.9).unwrap()
                * (F::one() / error_norm).powf(F::from_f64(1.0 / 5.0).unwrap());
            h *= factor
                .max(F::from_f64(0.1).unwrap())
                .min(F::from_f64(0.5).unwrap());
            rejected_steps += 1;
        }
    }

    // Check if integration was successful
    let success = t >= t_end;
    let message = if success {
        Some(format!("Integration successful, reached t = {:?}", t))
    } else {
        Some(format!("Integration incomplete, stopped at t = {:?}", t))
    };

    // Create dense output if requested
    let _dense_output = if opts.dense_output {
        Some(DenseSolution::new(
            t_values.clone(),
            y_values.clone(),
            Some(dy_values),
            Some(ContinuousOutputMethod::CubicHermite),
            None,
        ))
    } else {
        None
    };

    // Create result
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
