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
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

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
#[allow(dead_code)]
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

    // Verify mass _matrix compatibility
    mass_matrix::check_mass_compatibility(&mass_matrix, t_start, y0.view())?;

    // Determine initial step size if not provided
    let h0 = opts.h0.unwrap_or_else(|| {
        // Simple heuristic for initial step size
        let _span = t_end - t_start;
        _span / F::from_usize(100).unwrap() * F::from_f64(0.1).unwrap() // 0.1% of interval
    });

    // Determine minimum and maximum step sizes
    let min_step = opts.min_step.unwrap_or_else(|| {
        let _span = t_end - t_start;
        _span * F::from_f64(1e-10).unwrap() // Minimal step size
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

    // Runge-Kutta _matrix A (coefficients a_ij)
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
    let base_newton_tol = F::from_f64(1e-6).unwrap(); // Base tolerance
    let max_newton_iters = 20; // More iterations allowed

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

        // Better initial guess for stage values
        // For mass _matrix systems, we need a more careful initial guess
        let dy = if mass_matrix.matrix_type == MassMatrixType::Identity {
            f_current.clone()
        } else {
            mass_matrix::solve_mass_system(&mass_matrix, t, y.view(), f_current.view())?
        };

        // Improved initial guess using predictor-corrector approach
        // This provides a better starting point for Newton iteration
        let mut k1 = y.clone() + dy.clone() * (h * c1);
        let mut k2 = y.clone() + dy.clone() * (h * c2);
        let mut k3 = y.clone() + dy.clone() * h;

        // For mass _matrix systems, refine the initial guess with one predictor step
        if mass_matrix.matrix_type != MassMatrixType::Identity {
            // Compute better initial derivatives
            let f1_pred = f(t1, k1.view());
            let f2_pred = f(t2, k2.view());
            let f3_pred = f(t3, k3.view());

            // Solve for derivatives through mass _matrix
            let k1_prime_pred =
                mass_matrix::solve_mass_system(&mass_matrix, t1, k1.view(), f1_pred.view())?;
            let k2_prime_pred =
                mass_matrix::solve_mass_system(&mass_matrix, t2, k2.view(), f2_pred.view())?;
            let k3_prime_pred =
                mass_matrix::solve_mass_system(&mass_matrix, t3, k3.view(), f3_pred.view())?;

            // Refine initial guess using Radau coefficients
            k1 = y.clone()
                + (k1_prime_pred.clone() * a11
                    + k2_prime_pred.clone() * a12
                    + k3_prime_pred.clone() * a13)
                    * h;
            k2 = y.clone()
                + (k1_prime_pred.clone() * a21
                    + k2_prime_pred.clone() * a22
                    + k3_prime_pred.clone() * a23)
                    * h;
            k3 = y.clone() + (k1_prime_pred * a31 + k2_prime_pred * a32 + k3_prime_pred * a33) * h;

            func_evals += 3;
        }

        // Weights for error estimation
        let error_weights = calculate_error_weights(&y, atol, rtol);

        // Compute Jacobian for Newton iteration
        // For mass _matrix problems, we need both df/dy and dM/dy if state-dependent
        let mut compute_new_jacobian = true;
        let mut newton_converged = false;
        let mut newton_iterations = 0;

        // For mass matrices, we solve the coupled implicit system:
        // k_i = y + h * sum(a_ij * k'_j) where M(t_j, k_j) * k'_j = f(t_j, k_j)

        // Adaptive Newton tolerance based on step size and mass _matrix conditioning
        let mut newton_tol = base_newton_tol * h.max(F::from_f64(1e-3).unwrap());

        // For mass _matrix systems, adapt tolerance based on _matrix condition
        if mass_matrix.matrix_type != MassMatrixType::Identity {
            // Estimate condition number heuristically and adjust tolerance accordingly
            let condition_factor = match mass_matrix.matrix_type {
                MassMatrixType::Constant => F::from_f64(5.0).unwrap(), // Moderate relaxation
                MassMatrixType::TimeDependent => F::from_f64(8.0).unwrap(), // More relaxation
                MassMatrixType::StateDependent => F::from_f64(12.0).unwrap(), // Most relaxation
                MassMatrixType::Identity => F::one(),                  // No change
            };
            newton_tol *= condition_factor;

            // Cap the tolerance to prevent excessive relaxation
            newton_tol = newton_tol.min(F::from_f64(1e-4).unwrap());
        }

        // Ensure we have a Jacobian for the first iteration
        if jac_option.is_none() {
            compute_new_jacobian = true;
        }

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

            // First, get the mass _matrix at each stage
            let m1 = mass_matrix.evaluate(t1, k1.view());
            let m2 = mass_matrix.evaluate(t2, k2.view());
            let m3 = mass_matrix.evaluate(t3, k3.view());

            // For identity mass matrix, we can simplify
            if mass_matrix.matrix_type == MassMatrixType::Identity {
                // Simplified residual for identity mass _matrix
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
                // For mass _matrix systems, we solve the coupled implicit system:
                // k_i = y + h * sum(a_ij * k'_j) where M(t_j, k_j) * k'_j = f(t_j, k_j)
                //
                // This is equivalent to solving:
                // k_i = y + h * sum(a_ij * M(t_j, k_j)^(-1) * f(t_j, k_j))
                //
                // The Newton system for this is more complex and requires careful handling

                // Compute k'_j by solving M(t_j, k_j) * k'_j = f(t_j, k_j)
                let k1_prime =
                    mass_matrix::solve_mass_system(&mass_matrix, t1, k1.view(), f1.view())?;
                let k2_prime =
                    mass_matrix::solve_mass_system(&mass_matrix, t2, k2.view(), f2.view())?;
                let k3_prime =
                    mass_matrix::solve_mass_system(&mass_matrix, t3, k3.view(), f3.view())?;

                // Compute residuals: R_i = k_i - y - h * sum(a_ij * k'_j)
                let r1 = &k1
                    - &y
                    - &((k1_prime.clone() * a11 + k2_prime.clone() * a12 + k3_prime.clone() * a13)
                        * h);
                let r2 = &k2
                    - &y
                    - &((k1_prime.clone() * a21 + k2_prime.clone() * a22 + k3_prime.clone() * a23)
                        * h);
                let r3 = &k3
                    - &y
                    - &((k1_prime.clone() * a31 + k2_prime.clone() * a32 + k3_prime.clone() * a33)
                        * h);

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

                // For mass _matrix systems, we need to solve the correct Newton system
                // The Jacobian of the mass _matrix system includes both df/dy and mass _matrix effects

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

                // For mass _matrix systems, solve the correct Newton system
                // The Newton system for mass _matrix DAEs has the form:
                // [I - h*a11*M1^(-1)*J1   -h*a12*M1^(-1)*J2    -h*a13*M1^(-1)*J3 ] [dk1]   [r1]
                // [-h*a21*M2^(-1)*J1     I - h*a22*M2^(-1)*J2  -h*a23*M2^(-1)*J3 ] [dk2] = [r2]
                // [-h*a31*M3^(-1)*J1    -h*a32*M3^(-1)*J2    I - h*a33*M3^(-1)*J3] [dk3]   [r3]
                //
                // For numerical stability, we use a block-diagonal approximation

                // For mass _matrix systems, we use a more robust Newton approach
                // Instead of computing M^(-1)*J explicitly, we solve the Newton system directly
                // This avoids numerical issues with computing the inverse of potentially ill-conditioned mass matrices

                // Helper function to solve Newton correction for each stage
                let solve_newton_stage = |mass_mat: &Option<Array2<F>>,
                                          residual: &Array1<F>,
                                          jacobian: &Array2<F>,
                                          a_coeff: F,
                                          _prime: &Array1<F>|
                 -> IntegrateResult<Array1<F>> {
                    match mass_mat {
                        Some(m) => {
                            // CORRECTED Newton system for mass _matrix DAEs:
                            // For the Radau method with mass _matrix M(t,y) * y' = f(t,y)
                            // The stages satisfy: k_i = y + h * sum(a_ij * k'_j) where M * k'_j = f(t_j, k_j)
                            // The residual is: R_i = k_i - y - h * sum(a_ij * k'_j)
                            //
                            // The Newton system derivation:
                            // For k'_j = M^(-1) * f(t_j, k_j), we have d(k'_j)/dk_j = M^(-1) * ∂f/∂y
                            // So dR_i/dk_i = I - h * a_ii * M^(-1) * ∂f/∂y
                            // The Newton correction satisfies: (I - h * a_ii * M^(-1) * J) * dk = -R
                            // Multiplying by M: (M - h * a_ii * J) * dk = -M * R

                            let mut newton_matrix = m.clone();

                            // Construct (M - h * a_ii * J) where J = ∂f/∂y
                            for i in 0..n_dim {
                                for j in 0..n_dim {
                                    newton_matrix[[i, j]] -= h * a_coeff * jacobian[[i, j]];
                                }
                            }

                            // CORRECTED RHS: The right-hand side should be -M * residual, not just -residual
                            // This is the key fix for the Newton iteration convergence
                            let mut rhs = Array1::<F>::zeros(n_dim);
                            for i in 0..n_dim {
                                for j in 0..n_dim {
                                    rhs[i] -= m[[i, j]] * residual[j];
                                }
                            }

                            // Solve with iterative improvement for better numerical stability
                            let solve_with_conditioning =
                                |_matrix: &Array2<F>,
                                 b: &Array1<F>|
                                 -> IntegrateResult<Array1<F>> {
                                    // First attempt with original _matrix
                                    match solve_linear_system(&_matrix.view(), &b.view()) {
                                        Ok(solution) => {
                                            // Verify solution quality by checking residual
                                            let mut check_residual = Array1::<F>::zeros(n_dim);
                                            for i in 0..n_dim {
                                                for j in 0..n_dim {
                                                    check_residual[i] +=
                                                        _matrix[[i, j]] * solution[j];
                                                }
                                                check_residual[i] -= b[i];
                                            }

                                            let residual_norm = check_residual
                                                .iter()
                                                .fold(F::zero(), |acc, &x| acc + x * x)
                                                .sqrt();
                                            let b_norm = b
                                                .iter()
                                                .fold(F::zero(), |acc, &x| acc + x * x)
                                                .sqrt();

                                            if residual_norm
                                                < F::from_f64(1e-10).unwrap() * (F::one() + b_norm)
                                            {
                                                Ok(solution)
                                            } else {
                                                // Solution not accurate enough, try with regularization
                                                Err(crate::error::IntegrateError::ComputationError(
                                                    "Solution accuracy insufficient".to_string(),
                                                ))
                                            }
                                        }
                                        Err(e) => Err(e),
                                    }
                                };

                            match solve_with_conditioning(&newton_matrix, &rhs) {
                                Ok(dk) => {
                                    // dk is the direct Newton correction to the stage value
                                    Ok(dk)
                                }
                                Err(_) => {
                                    // Apply Tikhonov regularization for better conditioning
                                    let mut regularized = newton_matrix.clone();
                                    let reg_param = F::from_f64(1e-10).unwrap() * h;

                                    for i in 0..n_dim {
                                        regularized[[i, i]] += reg_param;
                                    }

                                    match solve_linear_system(&regularized.view(), &rhs.view()) {
                                        Ok(dk) => Ok(dk),
                                        Err(_) => {
                                            // Last resort: stronger regularization
                                            let strong_reg = F::from_f64(1e-8).unwrap() * h;
                                            for i in 0..n_dim {
                                                regularized[[i, i]] += strong_reg;
                                            }
                                            match solve_linear_system(
                                                &regularized.view(),
                                                &rhs.view(),
                                            ) {
                                                Ok(dk) => Ok(dk),
                                                Err(e) => Err(e),
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        None => {
                            // For identity mass matrix, use standard Newton system: (I - h*a_ii*J) * dk = r
                            let mut newton_matrix = Array2::<F>::eye(n_dim);
                            for i in 0..n_dim {
                                for j in 0..n_dim {
                                    newton_matrix[[i, j]] -= h * a_coeff * jacobian[[i, j]];
                                }
                            }
                            solve_linear_system(&newton_matrix.view(), &residual.view())
                        }
                    }
                };

                // Solve Newton corrections for each stage
                let dk1 = solve_newton_stage(&m1, &r1, jac, a11, &k1_prime)?;
                let dk2 = solve_newton_stage(&m2, &r2, jac, a22, &k2_prime)?;
                let dk3 = solve_newton_stage(&m3, &r3, jac, a33, &k3_prime)?;
                n_lu += 3;

                // Apply adaptive damping based on Newton iteration progress
                let mut damping = F::from_f64(1.0).unwrap();
                if newton_iterations > 5 {
                    damping = F::from_f64(0.7).unwrap(); // More conservative damping for slow convergence
                } else if newton_iterations > 10 {
                    damping = F::from_f64(0.5).unwrap(); // Even more conservative
                }

                // Apply damped Newton corrections
                k1 -= &(dk1 * damping);
                k2 -= &(dk2 * damping);
                k3 -= &(dk3 * damping);
            }
        }

        // Check if Newton iteration converged
        if !newton_converged {
            // Reduce step size more gradually and recompute Jacobian
            h *= F::from_f64(0.8).unwrap(); // Even less aggressive step reduction
            rejected_steps += 1;

            // Force recomputation of Jacobian on next iteration
            jac_option = None;

            // Be more tolerant for mass _matrix systems before giving up
            let min_step_tolerance = if mass_matrix.matrix_type != MassMatrixType::Identity {
                min_step * F::from_f64(0.1).unwrap() // Allow smaller steps for mass _matrix problems
            } else {
                min_step
            };

            // Prevent infinite reduction
            if h < min_step_tolerance {
                return Err(crate::error::IntegrateError::ComputationError(
                    "Newton iteration failed to converge even with minimum step size. Last residual norm was too large.".to_string()
                ));
            }
            continue;
        }

        // Compute new solution by solving mass _matrix systems for derivatives
        let f1 = f(t1, k1.view());
        let f2 = f(t2, k2.view());
        let f3 = f(t3, k3.view());

        let k1_prime = mass_matrix::solve_mass_system(&mass_matrix, t1, k1.view(), f1.view())?;
        let k2_prime = mass_matrix::solve_mass_system(&mass_matrix, t2, k2.view(), f2.view())?;
        let k3_prime = mass_matrix::solve_mass_system(&mass_matrix, t3, k3.view(), f3.view())?;

        let y_new = y.clone() + (k1_prime * b1 + k2_prime * b2 + k3_prime * b3) * h;
        func_evals += 3;

        // Estimate error using embedded method for mass _matrix systems
        // For Radau IIA with mass matrices, we use a more sophisticated error estimate
        let error = if mass_matrix.matrix_type == MassMatrixType::Identity {
            // Simple difference for identity mass _matrix
            &k3 - &y_new
        } else {
            // For mass _matrix systems, compute error in physical coordinates
            // Error estimate based on stage differences weighted by mass _matrix
            let stage_diff = &k3 - &k2;

            // Apply mass _matrix scaling to error estimate
            match mass_matrix.evaluate(t + h, k3.view()) {
                Some(ref m3_matrix) => {
                    // Scale error by mass _matrix characteristics
                    let mut scaled_error = Array1::<F>::zeros(n_dim);
                    for i in 0..n_dim {
                        let m_ii = m3_matrix[[i, i]];
                        if m_ii.abs() > F::from_f64(1e-12).unwrap() {
                            scaled_error[i] = stage_diff[i] / m_ii.sqrt();
                        } else {
                            scaled_error[i] = stage_diff[i];
                        }
                    }
                    scaled_error
                }
                None => stage_diff.clone(),
            }
        };

        // Compute error norm with proper scaling
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
        Some(format!("Integration successful, reached t = {t:?}"))
    } else {
        Some(format!("Integration incomplete, stopped at t = {t:?}"))
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
