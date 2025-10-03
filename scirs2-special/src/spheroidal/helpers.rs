//! Helper functions for spheroidal wave function computations
//!
//! This module provides utility functions used by both prolate and oblate spheroidal
//! function implementations, including Legendre function derivatives, eigenvalue solvers,
//! and numerical utilities.

use crate::error::{SpecialError, SpecialResult};

pub fn compute_legendre_assoc_derivative(n: usize, m: i32, x: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }

    // Handle boundary cases x = ±1
    if x.abs() >= 1.0 - 1e-10 {
        if m == 0 {
            // For P_n^0(x), the derivative at x = ±1 is n(n+1)/2 * (±1)^{n+1}
            let sign = if x > 0.0 {
                if n.is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                }
            } else if n.is_multiple_of(2) {
                -1.0
            } else {
                1.0
            };
            return (n as f64) * (n as f64 + 1.0) / 2.0 * sign;
        } else if m == 1 {
            // For P_n^1(x), derivative at x = ±1 involves more complex formulas
            let sign = if x > 0.0 { 1.0 } else { -1.0 };
            return sign * (n as f64) * (n as f64 + 1.0) * (n as f64) * (n as f64 - 1.0) / 4.0;
        } else {
            // Higher orders are zero at x = ±1 for m > 1
            return 0.0;
        }
    }

    // Use the standard recurrence relation for |x| < 1
    let p_n = crate::orthogonal::legendre_assoc(n, m, x);

    if n == 0 {
        return 0.0;
    }

    let p_nminus_1 = crate::orthogonal::legendre_assoc(n - 1, m, x);
    let n_f64 = n as f64;
    let m_f64 = m as f64;

    // Standard derivative formula
    let numerator = n_f64 * x * p_n - (n_f64 + m_f64) * p_nminus_1;
    let denominator = x * x - 1.0;

    if denominator.abs() < 1e-10 {
        // Near x = ±1, use L'Hôpital's rule or series expansion
        // This is a simplified approximation
        return n_f64 * (n_f64 + 1.0) / 2.0 * x.signum();
    }

    numerator / denominator
}

/// Compute the associated Legendre functions of the second kind Q_n^m(x) and their derivatives
///
/// The associated Legendre functions of the second kind are defined for |x| > 1 and satisfy
/// the same differential equation as P_n^m(x) but have different boundary conditions.
///
/// # Arguments
///
/// * `n` - Degree (≥ 0)
/// * `m` - Order (|m| ≤ n)
/// * `x` - Argument (|x| > 1)
///
/// # Returns
///
/// * `SpecialResult<(f64, f64)>` - (Q_n^m(x), dQ_n^m(x)/dx)
#[allow(dead_code)]
pub fn legendre_associated_second_kind(n: i32, m: i32, x: f64) -> SpecialResult<(f64, f64)> {
    // Parameter validation
    if n < 0 || m.abs() > n {
        return Err(SpecialError::DomainError(format!(
            "Invalid parameters: n={n}, m={m}, must have n≥0 and |m|≤n"
        )));
    }

    if x.abs() <= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "Argument x={x} must satisfy |x| > 1"
        )));
    }

    // Handle special cases
    if n == 0 && m == 0 {
        // Q_0^0(x) = (1/2) * ln((x+1)/(x-1))
        let q_00 = 0.5 * ((x + 1.0) / (x - 1.0)).ln();
        let q_00_prime = -1.0 / (x * x - 1.0);
        return Ok((q_00, q_00_prime));
    }

    // For general case, use recurrence relations and analytical expressions
    if m == 0 {
        legendre_second_kind_m0(n, x)
    } else {
        legendre_second_kind_general(n, m, x)
    }
}

/// Compute Q_n^0(x) using recurrence relations and analytical expressions
#[allow(dead_code)]
fn legendre_second_kind_m0(n: i32, x: f64) -> SpecialResult<(f64, f64)> {
    if n == 0 {
        let q_0 = 0.5 * ((x + 1.0) / (x - 1.0)).ln();
        let q_0_prime = -1.0 / (x * x - 1.0);
        return Ok((q_0, q_0_prime));
    }

    if n == 1 {
        let q_0 = 0.5 * ((x + 1.0) / (x - 1.0)).ln();
        let q_1 = x * q_0 - 1.0;
        let q_1_prime = q_0 + x / (x * x - 1.0);
        return Ok((q_1, q_1_prime));
    }

    // Use the recurrence relation: (n+1)Q_{n+1}(x) = (2n+1)xQ_n(x) - nQ_{n-1}(x)
    let mut q_prev = 0.5 * ((x + 1.0) / (x - 1.0)).ln(); // Q_0
    let mut q_curr = x * q_prev - 1.0; // Q_1

    for k in 2..=n {
        let k_f = k as f64;
        let q_next = ((2.0 * k_f - 1.0) * x * q_curr - (k_f - 1.0) * q_prev) / k_f;
        q_prev = q_curr;
        q_curr = q_next;
    }

    // Compute derivative using the recurrence relation for derivatives
    // Q'_n(x) = n * [x * Q_n(x) - Q_{n-1}(x)] / (x^2 - 1)
    let q_n_prime = if n == 1 {
        let q_0 = 0.5 * ((x + 1.0) / (x - 1.0)).ln();
        q_0 + x / (x * x - 1.0)
    } else {
        let n_f = n as f64;
        n_f * (x * q_curr - q_prev) / (x * x - 1.0)
    };

    Ok((q_curr, q_n_prime))
}

/// Compute Q_n^m(x) for general m using the relationship with P_n^m and Q_n^0
#[allow(dead_code)]
fn legendre_second_kind_general(n: i32, m: i32, x: f64) -> SpecialResult<(f64, f64)> {
    let m_abs = m.abs();

    // Use the formula: Q_n^m(x) = (-1)^m * (1-x^2)^{m/2} * d^m/dx^m Q_n(x)
    // For computational efficiency, use the recurrence relation:
    // Q_n^m(x) = (x^2-1)^{1/2} * [m*x/(x^2-1) * Q_n^{m-1}(x) + Q_n^{m-1}'(x)]

    if m_abs == 0 {
        return legendre_second_kind_m0(n, x);
    }

    // Start with Q_n^0
    let (mut q_nm, mut q_nm_prime) = legendre_second_kind_m0(n, x)?;
    let sqrt_x2minus_1 = (x * x - 1.0).sqrt();

    // Apply the recurrence relation to build up to Q_n^m
    for k in 1..=m_abs {
        let k_f = k as f64;
        let x2minus_1 = x * x - 1.0;

        // Q_n^k(x) = sqrt(x^2-1) * [k*x/(x^2-1) * Q_n^{k-1}(x) + Q_n^{k-1}'(x)]
        let new_q_nm = sqrt_x2minus_1 * (k_f * x / x2minus_1 * q_nm + q_nm_prime);

        // Derivative: d/dx Q_n^k(x)
        let new_q_nm_prime = x / sqrt_x2minus_1 * (k_f * x / x2minus_1 * q_nm + q_nm_prime)
            + sqrt_x2minus_1
                * (
                    k_f / x2minus_1 * q_nm + k_f * x / x2minus_1 * q_nm_prime
                        - 2.0 * k_f * x * x / (x2minus_1 * x2minus_1) * q_nm
                        + q_nm_prime
                    // derivative of Q_n^{k-1}'
                );

        q_nm = new_q_nm;
        q_nm_prime = new_q_nm_prime;
    }

    // Apply the (-1)^m factor for negative m
    if m < 0 {
        let sign = if m_abs % 2 == 0 { 1.0 } else { -1.0 };
        // Also need to apply the factor (-1)^m * (n-m)! / (n+m)!
        use crate::combinatorial::factorial;
        let factor = sign * factorial((n - m_abs) as u32).unwrap_or(1.0)
            / factorial((n + m_abs) as u32).unwrap_or(1.0);
        q_nm *= factor;
        q_nm_prime *= factor;
    }

    Ok((q_nm, q_nm_prime))
}

/// Improved eigenvalue solver for spheroidal wave functions
///
/// Solves the characteristic equation using the three-term recurrence relation
/// and matrix eigenvalue methods for better accuracy
#[allow(dead_code)]
pub fn solve_spheroidal_eigenvalue_improved(m: i32, n: i32, c: f64) -> SpecialResult<f64> {
    // For small c, use perturbation theory
    if c.abs() < 0.1 {
        return solve_eigenvalue_perturbation(m, n, c);
    }

    // For larger c, use matrix methods to solve the infinite system
    let matrixsize = (4 * n.abs() + 20).min(100) as usize;
    solve_eigenvalue_matrix_method(m, n, c, matrixsize)
}

/// Solve eigenvalue using perturbation theory for small c
#[allow(dead_code)]
fn solve_eigenvalue_perturbation(m: i32, n: i32, c: f64) -> SpecialResult<f64> {
    let _m_f = m as f64;
    let n_f = n as f64;

    // Leading term
    let lambda_0 = n_f * (n_f + 1.0);

    // First-order correction
    let lambda_1 = if n == 0 && m == 0 {
        c * c / 2.0
    } else {
        let factor = if n % 2 == m % 2 { 1.0 } else { -1.0 };
        factor * c * c / (2.0 * (2.0 * n_f + 1.0))
    };

    // Second-order correction (more complex, approximated)
    let lambda_2 =
        -c.powi(4) / (8.0 * (2.0 * n_f + 1.0) * (2.0 * n_f + 3.0) * (2.0 * n_f - 1.0).max(1.0));

    Ok(lambda_0 + lambda_1 + lambda_2)
}

/// Solve eigenvalue using matrix methods for the three-term recurrence
#[allow(dead_code)]
fn solve_eigenvalue_matrix_method(m: i32, n: i32, c: f64, matrixsize: usize) -> SpecialResult<f64> {
    // Set up the tridiagonal matrix for the recurrence relation
    // (α_r - λ)a_r + β_{r+1}a_{r+2} + β_{r-1}a_{r-2} = 0

    let m_f = m as f64;
    let c2 = c * c;

    // Create the coefficient matrix
    let mut main_diag = vec![0.0; matrixsize];
    let mut upper_diag = vec![0.0; matrixsize - 1];
    let mut lower_diag = vec![0.0; matrixsize - 1];

    for i in 0..matrixsize {
        let r = (i as i32 + m - n).abs();
        let r_f = r as f64;

        // α_r = r(r+1) for the base problem
        main_diag[i] = (r_f + m_f) * (r_f + m_f + 1.0);

        // β_r coefficients
        if i < matrixsize - 1 {
            let beta = c2 / (4.0 * (2.0 * r_f + 1.0) * (2.0 * r_f + 3.0));
            upper_diag[i] = beta;
        }

        if i > 0 {
            let r_prev = ((i - 1) as i32 + m - n).abs() as f64;
            let beta = c2 / (4.0 * (2.0 * r_prev + 1.0) * (2.0 * r_prev + 3.0));
            lower_diag[i - 1] = beta;
        }
    }

    // Use proper eigenvalue solver with inverse iteration for the target eigenvalue
    solve_tridiagonal_eigenvalue(&main_diag, &upper_diag, n)
}

/// Solve for a specific eigenvalue of a symmetric tridiagonal matrix using inverse iteration
/// This finds the eigenvalue closest to n(n+1), which is the expected characteristic value
fn solve_tridiagonal_eigenvalue(
    main_diag: &[f64],
    off_diag: &[f64],
    target_n: i32,
) -> SpecialResult<f64> {
    let n = main_diag.len();
    if n == 0 {
        return Err(SpecialError::ComputationError("Empty matrix".to_string()));
    }

    // Initial guess for eigenvalue near n(n+1)
    let target_f = target_n as f64;
    let mut sigma = target_f * (target_f + 1.0);

    let max_iter = 100;
    let tolerance = 1e-12;

    // Inverse iteration with Rayleigh quotient shift
    let mut v = vec![1.0; n];
    // Normalize initial vector
    let mut norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for vi in v.iter_mut() {
        *vi /= norm;
    }

    for _ in 0..max_iter {
        // Solve (A - σI)w = v using tridiagonal solver
        let w = solve_shifted_tridiagonal(main_diag, off_diag, &v, sigma)?;

        // Normalize w
        norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-100 {
            return Err(SpecialError::ComputationError(
                "Eigenvalue iteration produced zero vector".to_string(),
            ));
        }

        for i in 0..n {
            v[i] = w[i] / norm;
        }

        // Compute Rayleigh quotient: λ = v^T A v
        let mut lambda = 0.0;
        for i in 0..n {
            lambda += v[i] * main_diag[i] * v[i];
            if i < n - 1 {
                lambda += 2.0 * v[i] * off_diag[i] * v[i + 1];
            }
        }

        // Check convergence
        if (lambda - sigma).abs() < tolerance {
            return Ok(lambda);
        }

        // Update shift
        sigma = lambda;
    }

    // If not converged, return best approximation
    Ok(sigma)
}

/// Solve the tridiagonal system (A - σI)x = b using Thomas algorithm
fn solve_shifted_tridiagonal(
    main_diag: &[f64],
    off_diag: &[f64],
    rhs: &[f64],
    sigma: f64,
) -> SpecialResult<Vec<f64>> {
    let n = main_diag.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Create shifted diagonal
    let mut a = vec![0.0; n]; // lower diagonal
    let mut b = vec![0.0; n]; // main diagonal (shifted)
    let mut c = vec![0.0; n]; // upper diagonal

    for i in 0..n {
        b[i] = main_diag[i] - sigma;
        if i < n - 1 {
            c[i] = off_diag[i];
            a[i + 1] = off_diag[i];
        }
    }

    // Thomas algorithm (forward elimination)
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = c[0] / b[0];
    d_prime[0] = rhs[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i] * c_prime[i - 1];
        if denom.abs() < 1e-100 {
            // Matrix is nearly singular, use regularization
            let reg_denom = denom + 1e-10 * denom.signum();
            c_prime[i] = if i < n - 1 { c[i] / reg_denom } else { 0.0 };
            d_prime[i] = (rhs[i] - a[i] * d_prime[i - 1]) / reg_denom;
        } else {
            c_prime[i] = if i < n - 1 { c[i] / denom } else { 0.0 };
            d_prime[i] = (rhs[i] - a[i] * d_prime[i - 1]) / denom;
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}
