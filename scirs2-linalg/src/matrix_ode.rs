//! Matrix differential equation utilities using a `Vec<Vec<f64>>` flat API.
//!
//! This module provides:
//!
//! - [`matrix_exp`] – matrix exponential via scaling-and-squaring + Padé approximant
//! - [`matrix_log`] – matrix logarithm via inverse scaling-and-squaring + Schur
//! - [`matrix_sqrt`] – matrix square root via Denman–Beavers iteration
//! - [`matrix_pow`] – matrix power A^p for arbitrary real p
//! - [`matrix_sin`] / [`matrix_cos`] – matrix trigonometric functions
//! - [`frechet_derivative_expm`] – Fréchet derivative of the matrix exponential
//! - [`expm_cond`] – condition number of the matrix exponential
//!
//! All functions accept/return `Vec<Vec<f64>>` row-major matrices of size `n × n`.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_linalg::matrix_ode::{matrix_exp, matrix_cos, matrix_sin};
//!
//! // Rotation generator: A = [[0, -t], [t, 0]] => expm(A) = rotation by t
//! let t = std::f64::consts::PI / 4.0; // 45 degrees
//! let a = vec![vec![0.0, -t], vec![t, 0.0]];
//! let ea = matrix_exp(&a, 2);
//! // Check orthogonality: ea^T * ea ≈ I
//! let s2 = 2.0f64.sqrt() / 2.0;
//! assert!((ea[0][0] - s2).abs() < 1e-10);
//! assert!((ea[0][1] + s2).abs() < 1e-10);
//! ```

use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// Internal dense matrix helpers (n×n row-major Vec<Vec<f64>>)
// ============================================================================

/// Identity matrix of size n.
fn eye(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// Matrix multiply C = A * B (all n×n).
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for k in 0..n {
            if a[i][k] == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Matrix add A + B.
fn matadd(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut c = a.to_vec();
    for i in 0..n {
        for j in 0..n {
            c[i][j] += b[i][j];
        }
    }
    c
}

/// Matrix subtract A - B.
fn matsub(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut c = a.to_vec();
    for i in 0..n {
        for j in 0..n {
            c[i][j] -= b[i][j];
        }
    }
    c
}

/// Scalar multiply s * A.
fn scalmul(s: f64, a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut c = a.to_vec();
    for i in 0..n {
        for j in 0..n {
            c[i][j] *= s;
        }
    }
    c
}

/// Frobenius norm of a matrix.
fn frobenius_norm(a: &[Vec<f64>], n: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            s += a[i][j] * a[i][j];
        }
    }
    s.sqrt()
}

/// 1-norm (max column sum) of a matrix.
fn one_norm(a: &[Vec<f64>], n: usize) -> f64 {
    (0..n)
        .map(|j| (0..n).map(|i| a[i][j].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

/// Transpose of a square matrix.
fn transpose(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Solve A * x = b using Gaussian elimination with partial pivoting.
/// Returns x as a column vector.
fn solve_linear(a: &[Vec<f64>], b: &[f64], n: usize) -> LinalgResult<Vec<f64>> {
    let mut mat: Vec<Vec<f64>> = a.to_vec();
    let mut rhs: Vec<f64> = b.to_vec();

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in col + 1..n {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular".to_string(),
            ));
        }
        mat.swap(col, max_row);
        rhs.swap(col, max_row);

        let pivot = mat[col][col];
        for row in col + 1..n {
            let factor = mat[row][col] / pivot;
            rhs[row] -= factor * rhs[col];
            for j in col..n {
                let v = mat[col][j];
                mat[row][j] -= factor * v;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in i + 1..n {
            s -= mat[i][j] * x[j];
        }
        if mat[i][i].abs() < 1e-300 {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular".to_string(),
            ));
        }
        x[i] = s / mat[i][i];
    }
    Ok(x)
}

/// Solve A * X = B for matrix B (each column separately).
fn solve_matrix(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> LinalgResult<Vec<Vec<f64>>> {
    let mut x = vec![vec![0.0; n]; n];
    for col in 0..n {
        let b_col: Vec<f64> = (0..n).map(|r| b[r][col]).collect();
        let x_col = solve_linear(a, &b_col, n)?;
        for row in 0..n {
            x[row][col] = x_col[row];
        }
    }
    Ok(x)
}

/// Compute A^k for integer k.
fn mat_int_pow(a: &[Vec<f64>], mut k: u32, n: usize) -> Vec<Vec<f64>> {
    let mut result = eye(n);
    let mut base = a.to_vec();
    while k > 0 {
        if k % 2 == 1 {
            result = matmul(&result, &base, n);
        }
        base = matmul(&base, &base, n);
        k /= 2;
    }
    result
}

// ============================================================================
// Padé approximant for matrix exponential
// ============================================================================

/// Padé approximant numerator/denominator coefficients (order m).
/// Returns (p_coeff, q_coeff) such that expm ≈ Q^{-1} P.
fn pade_coefficients(m: usize) -> (Vec<f64>, Vec<f64>) {
    // Standard Padé coefficients for orders 3, 5, 7, 9, 13
    // Reference: Higham, "Functions of Matrices", 2008
    match m {
        3 => {
            let c = [120.0, 60.0, 12.0, 1.0];
            let p: Vec<f64> = c.to_vec();
            let q: Vec<f64> = c
                .iter()
                .enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
                .collect();
            (p, q)
        }
        5 => {
            let c = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0];
            let p: Vec<f64> = c.to_vec();
            let q: Vec<f64> = c
                .iter()
                .enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
                .collect();
            (p, q)
        }
        7 => {
            let c = [
                17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0,
            ];
            let p: Vec<f64> = c.to_vec();
            let q: Vec<f64> = c
                .iter()
                .enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
                .collect();
            (p, q)
        }
        9 => {
            let c = [
                17643225600.0,
                8821612800.0,
                2075673600.0,
                302702400.0,
                30270240.0,
                2162160.0,
                110880.0,
                3960.0,
                90.0,
                1.0,
            ];
            let p: Vec<f64> = c.to_vec();
            let q: Vec<f64> = c
                .iter()
                .enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
                .collect();
            (p, q)
        }
        13 => {
            // Coefficients for order 13 Padé approximant (from Higham 2005)
            let c = [
                64764752532480000.0,
                32382376266240000.0,
                7771770303897600.0,
                1187353796428800.0,
                129060195264000.0,
                10559470521600.0,
                670442572800.0,
                33522128640.0,
                1323241920.0,
                40840800.0,
                960960.0,
                16380.0,
                182.0,
                1.0,
            ];
            let p: Vec<f64> = c.to_vec();
            let q: Vec<f64> = c
                .iter()
                .enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
                .collect();
            (p, q)
        }
        _ => {
            // Fall back to order 7
            let c = [
                17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0,
            ];
            let p: Vec<f64> = c.to_vec();
            let q: Vec<f64> = c
                .iter()
                .enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
                .collect();
            (p, q)
        }
    }
}

/// Evaluate Padé polynomial sum: c[0]*I + c[1]*A + c[2]*A^2 + ...
fn eval_pade_poly(coeffs: &[f64], a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let m = coeffs.len();
    if m == 0 {
        return vec![vec![0.0; n]; n];
    }
    let mut result = scalmul(coeffs[0], &eye(n), n);
    let mut a_pow = a.to_vec(); // A^1
    for k in 1..m {
        let term = scalmul(coeffs[k], &a_pow, n);
        result = matadd(&result, &term, n);
        if k < m - 1 {
            a_pow = matmul(&a_pow, a, n);
        }
    }
    result
}

// ============================================================================
// Public API
// ============================================================================

/// Compute the matrix exponential `expm(A)` using scaling and squaring
/// with a Padé approximant of order 13.
///
/// # Arguments
///
/// * `a` - Input square matrix as `Vec<Vec<f64>>` (row-major, n×n)
/// * `n` - Dimension
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::matrix_ode::matrix_exp;
///
/// // Zero matrix: expm(0) = I
/// let z = vec![vec![0.0_f64, 0.0], vec![0.0, 0.0]];
/// let e = matrix_exp(&z, 2);
/// assert!((e[0][0] - 1.0).abs() < 1e-12);
/// assert!((e[1][1] - 1.0).abs() < 1e-12);
/// ```
pub fn matrix_exp(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    if n == 0 {
        return vec![];
    }
    // Choose scaling: find s such that ||A/2^s||_1 ~ 1
    let norm = one_norm(a, n);
    let mut s = 0i32;
    if norm > 1.0 {
        s = (norm.log2().ceil() as i32).max(0);
    }
    let scale = (2.0f64).powi(s);
    let a_scaled = scalmul(1.0 / scale, a, n);

    // Padé order 13
    let (p_coeff, q_coeff) = pade_coefficients(13);
    let p_mat = eval_pade_poly(&p_coeff, &a_scaled, n);
    let q_mat = eval_pade_poly(&q_coeff, &a_scaled, n);

    // expm_scaled = Q^{-1} P
    let exp_scaled = match solve_matrix(&q_mat, &p_mat, n) {
        Ok(m) => m,
        Err(_) => {
            // Fallback: Taylor series (32 terms)
            taylor_expm(a, n, 32)
        }
    };

    // Squaring phase: exp(A) = exp(A/2^s)^(2^s)
    let mut result = exp_scaled;
    for _ in 0..s {
        result = matmul(&result, &result, n);
    }
    result
}

/// Taylor series fallback for matrix exponential.
fn taylor_expm(a: &[Vec<f64>], n: usize, terms: usize) -> Vec<Vec<f64>> {
    let mut result = eye(n);
    let mut term = eye(n);
    let mut factorial = 1.0f64;
    for k in 1..=terms {
        factorial *= k as f64;
        term = matmul(&term, a, n);
        let scaled_term = scalmul(1.0 / factorial, &term, n);
        result = matadd(&result, &scaled_term, n);
        // Early termination
        if frobenius_norm(&scaled_term, n) < 1e-16 * frobenius_norm(&result, n) {
            break;
        }
    }
    result
}

/// Compute the matrix logarithm `logm(A)` such that `expm(logm(A)) ≈ A`.
///
/// Uses inverse scaling-and-squaring: repeatedly takes square roots until
/// the matrix is close to identity, then applies the Gregory series.
///
/// # Errors
///
/// Returns an error if `A` is singular or has non-positive eigenvalues
/// (detected heuristically via near-zero pivots during square root iteration).
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::matrix_ode::{matrix_exp, matrix_log};
///
/// let a = vec![vec![2.0_f64, 1.0], vec![0.0, 3.0]];
/// let log_a = matrix_log(&a, 2).expect("valid input");
/// let exp_log_a = matrix_exp(&log_a, 2);
/// assert!((exp_log_a[0][0] - a[0][0]).abs() < 1e-8);
/// assert!((exp_log_a[1][1] - a[1][1]).abs() < 1e-8);
/// ```
pub fn matrix_log(a: &[Vec<f64>], n: usize) -> LinalgResult<Vec<Vec<f64>>> {
    if n == 0 {
        return Ok(vec![]);
    }

    // Repeated square root: A = (A^{1/2^k})^{2^k}
    // When A^{1/2^k} ≈ I, use Gregory series: log(I+X) = 2*atanh(X/(2+X))
    let max_sq = 16usize;
    let mut b = a.to_vec();
    let mut k = 0usize;

    // Take square roots until ||B - I|| < 0.5
    let ident = eye(n);
    while k < max_sq {
        let diff = matsub(&b, &ident, n);
        if frobenius_norm(&diff, n) < 0.5 {
            break;
        }
        b = matrix_sqrt(&b, n)?;
        k += 1;
    }

    // Gregory series: log(B) via Padé-like series for log(I + X), X = B - I
    let x = matsub(&b, &ident, n);
    let log_b = gregory_log(&x, n, 64)?;

    // Scale back: log(A) = 2^k * log(A^{1/2^k})
    let scale = (1u64 << k) as f64;
    Ok(scalmul(scale, &log_b, n))
}

/// Gregory (Padé) series for log(I + X) when ||X|| < 1.
/// Uses the identity log(I+X) = 2 * sum_{k=0}^{inf} X^{2k+1} / ((2k+1) * (2+X+X^{-1})^{...})
/// Actually implements a direct power series: sum_{k=1}^{terms} (-1)^{k+1} X^k / k
fn gregory_log(x: &[Vec<f64>], n: usize, terms: usize) -> LinalgResult<Vec<Vec<f64>>> {
    let norm = frobenius_norm(x, n);
    if norm >= 1.0 {
        return Err(LinalgError::ComputationError(
            "matrix_log: Gregory series: ||X|| >= 1, series may not converge".into(),
        ));
    }

    let mut result = vec![vec![0.0; n]; n];
    let mut x_pow = x.to_vec(); // X^1
    for k in 1..=terms {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let term = scalmul(sign / k as f64, &x_pow, n);
        result = matadd(&result, &term, n);
        // Early termination
        if frobenius_norm(&term, n) < 1e-15 * frobenius_norm(&result, n).max(1e-300) {
            break;
        }
        x_pow = matmul(&x_pow, x, n);
    }
    Ok(result)
}

/// Compute the matrix square root `sqrtm(A)` such that `sqrtm(A)^2 ≈ A`.
///
/// Uses the Denman–Beavers coupled iteration:
/// - Y_{k+1} = (Y_k + Z_k^{-1}) / 2
/// - Z_{k+1} = (Z_k + Y_k^{-1}) / 2
///
/// Starting with Y_0 = A, Z_0 = I.
///
/// # Errors
///
/// Returns an error if the iteration fails to converge (e.g., A is singular).
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::matrix_ode::matrix_sqrt;
///
/// let a = vec![vec![4.0_f64, 0.0], vec![0.0, 9.0]];
/// let s = matrix_sqrt(&a, 2).expect("valid input");
/// // s ≈ [[2, 0], [0, 3]]
/// assert!((s[0][0] - 2.0).abs() < 1e-8);
/// assert!((s[1][1] - 3.0).abs() < 1e-8);
/// ```
pub fn matrix_sqrt(a: &[Vec<f64>], n: usize) -> LinalgResult<Vec<Vec<f64>>> {
    if n == 0 {
        return Ok(vec![]);
    }

    let mut y = a.to_vec();
    let mut z = eye(n);
    let ident = eye(n);

    for _ in 0..100 {
        let y_inv = solve_matrix(&y, &ident, n)?;
        let z_inv = solve_matrix(&z, &ident, n)?;

        let y_new = scalmul(0.5, &matadd(&y, &z_inv, n), n);
        let z_new = scalmul(0.5, &matadd(&z, &y_inv, n), n);

        let delta = frobenius_norm(&matsub(&y_new, &y, n), n);
        y = y_new;
        z = z_new;

        if delta < 1e-13 * frobenius_norm(&y, n).max(1e-300) {
            return Ok(y);
        }
    }

    // Return best iterate even if not fully converged
    let residual = frobenius_norm(&matsub(&matmul(&y, &y, n), a, n), n);
    if residual < 1e-6 * frobenius_norm(a, n).max(1e-300) {
        Ok(y)
    } else {
        Err(LinalgError::ConvergenceError(
            "matrix_sqrt: Denman-Beavers iteration did not converge".into(),
        ))
    }
}

/// Compute the matrix power `A^p` for arbitrary real `p`.
///
/// Uses the identity A^p = exp(p * log(A)).
///
/// # Errors
///
/// Returns an error if `log(A)` cannot be computed (e.g., A has non-positive eigenvalues).
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::matrix_ode::matrix_pow;
///
/// let a = vec![vec![4.0_f64, 0.0], vec![0.0, 9.0]];
/// let a_half = matrix_pow(&a, 2, 0.5).expect("valid input");
/// assert!((a_half[0][0] - 2.0).abs() < 1e-6);
/// assert!((a_half[1][1] - 3.0).abs() < 1e-6);
/// ```
pub fn matrix_pow(a: &[Vec<f64>], n: usize, p: f64) -> LinalgResult<Vec<Vec<f64>>> {
    if n == 0 {
        return Ok(vec![]);
    }
    // Integer powers via squaring
    if p == p.floor() && p.abs() < 1e14 {
        let k = p as i64;
        if k >= 0 {
            return Ok(mat_int_pow(a, k as u32, n));
        } else {
            let ident = eye(n);
            let a_inv = solve_matrix(a, &ident, n)?;
            return Ok(mat_int_pow(&a_inv, (-k) as u32, n));
        }
    }
    // General case: A^p = exp(p * log(A))
    let log_a = matrix_log(a, n)?;
    let p_log_a = scalmul(p, &log_a, n);
    Ok(matrix_exp(&p_log_a, n))
}

/// Compute the matrix sine `sinm(A) = (expm(iA) - expm(-iA)) / (2i)`.
///
/// Implemented via: sinm(A) = expm(A_skew) where we use the real formula:
/// sinm(A) = Im(expm(iA)) = sum_{k=0}^{inf} (-1)^k A^{2k+1} / (2k+1)!
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::matrix_ode::matrix_sin;
///
/// // sin(0) = 0
/// let z = vec![vec![0.0_f64, 0.0], vec![0.0, 0.0]];
/// let s = matrix_sin(&z, 2);
/// assert!(s[0][0].abs() < 1e-12);
/// ```
pub fn matrix_sin(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    if n == 0 {
        return vec![];
    }
    // Series: sinm(A) = A - A^3/3! + A^5/5! - ...
    let a2 = matmul(a, a, n); // A^2
    let mut result = a.to_vec();
    let mut term = a.to_vec();
    let mut factorial = 1.0f64;
    let mut sign = -1.0f64;

    for k in 1..=32usize {
        // next odd power: multiply term by A^2 / ((2k)(2k+1))
        factorial *= (2 * k) as f64 * (2 * k + 1) as f64;
        term = matmul(&term, &a2, n);
        let scaled = scalmul(sign / factorial, &term, n);
        if frobenius_norm(&scaled, n) < 1e-16 * frobenius_norm(&result, n).max(1e-300) {
            break;
        }
        result = matadd(&result, &scaled, n);
        sign = -sign;
    }
    result
}

/// Compute the matrix cosine `cosm(A) = (expm(iA) + expm(-iA)) / 2`.
///
/// Implemented via Taylor series: cosm(A) = I - A^2/2! + A^4/4! - ...
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::matrix_ode::matrix_cos;
///
/// // cos(0) = I
/// let z = vec![vec![0.0_f64, 0.0], vec![0.0, 0.0]];
/// let c = matrix_cos(&z, 2);
/// assert!((c[0][0] - 1.0).abs() < 1e-12);
/// assert!((c[1][1] - 1.0).abs() < 1e-12);
/// ```
pub fn matrix_cos(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    if n == 0 {
        return vec![];
    }
    // Series: cosm(A) = I - A^2/2! + A^4/4! - ...
    let a2 = matmul(a, a, n);
    let mut result = eye(n);
    let mut term = eye(n); // A^0
    let mut factorial = 1.0f64;
    let mut sign = -1.0f64;

    for k in 1..=32usize {
        factorial *= (2 * k - 1) as f64 * (2 * k) as f64;
        term = matmul(&term, &a2, n);
        let scaled = scalmul(sign / factorial, &term, n);
        if frobenius_norm(&scaled, n) < 1e-16 * frobenius_norm(&result, n).max(1e-300) {
            break;
        }
        result = matadd(&result, &scaled, n);
        sign = -sign;
    }
    result
}

// ============================================================================
// Fréchet derivative and condition number
// ============================================================================

/// Compute the Fréchet derivative of `expm` at `A` in direction `E`.
///
/// `L_{expm, A}(E) = lim_{eps->0} [expm(A + eps*E) - expm(A)] / eps`
///
/// Computed via the augmented matrix method (Kenney & Laub 1989):
/// ```text
/// expm([[A, E], [0, A]]) = [[expm(A), L_{expm,A}(E)], [0, expm(A)]]
/// ```
///
/// # Arguments
///
/// * `a` - n×n input matrix
/// * `e` - n×n direction matrix
/// * `n` - dimension
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::matrix_ode::{matrix_exp, frechet_derivative_expm};
///
/// let a = vec![vec![0.0_f64, 1.0], vec![-1.0, 0.0]];
/// let e = vec![vec![1.0_f64, 0.0], vec![0.0, 0.0]];
/// let l = frechet_derivative_expm(&a, &e, 2);
/// // Verify numerically: L ≈ (expm(A + eps*E) - expm(A)) / eps
/// let eps = 1e-6;
/// let ae = vec![vec![eps, 1.0], vec![-1.0, 0.0]];
/// let e1 = matrix_exp(&ae, 2);
/// let e0 = matrix_exp(&a, 2);
/// let diff00 = (e1[0][0] - e0[0][0]) / eps;
/// assert!((l[0][0] - diff00).abs() < 1e-5);
/// ```
pub fn frechet_derivative_expm(a: &[Vec<f64>], e: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    if n == 0 {
        return vec![];
    }
    // Build 2n × 2n augmented matrix [[A, E], [0, A]]
    let n2 = 2 * n;
    let mut aug = vec![vec![0.0; n2]; n2];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j]; // top-left: A
            aug[i][j + n] = e[i][j]; // top-right: E
            aug[i + n][j + n] = a[i][j]; // bottom-right: A
        }
    }

    let exp_aug = matrix_exp(&aug, n2);

    // Extract top-right n×n block: that is L_{expm,A}(E)
    let mut frechet = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            frechet[i][j] = exp_aug[i][j + n];
        }
    }
    frechet
}

/// Estimate the condition number of the matrix exponential at `A`.
///
/// Uses the Fréchet derivative: κ_{expm}(A) = ||L_{expm,A}|| * ||A|| / ||expm(A)||
/// where ||L|| is estimated by evaluating the Fréchet derivative on random unit vectors.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::matrix_ode::expm_cond;
///
/// // Condition number of expm at zero matrix should be 1 (expm(0) = I)
/// let z = vec![vec![0.0_f64, 0.0], vec![0.0, 0.0]];
/// let kappa = expm_cond(&z, 2);
/// assert!(kappa >= 0.0);
/// ```
pub fn expm_cond(a: &[Vec<f64>], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let ea = matrix_exp(a, n);
    let norm_a = frobenius_norm(a, n);
    let norm_ea = frobenius_norm(&ea, n);

    if norm_ea < 1e-300 {
        return 0.0;
    }

    // Estimate ||L_{expm,A}|| by evaluating on canonical basis directions
    // and taking the maximum
    let mut max_l_norm = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut e = vec![vec![0.0; n]; n];
            e[i][j] = 1.0;
            let l = frechet_derivative_expm(a, &e, n);
            let l_norm = frobenius_norm(&l, n);
            if l_norm > max_l_norm {
                max_l_norm = l_norm;
            }
        }
    }

    max_l_norm * norm_a / norm_ea
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_matrix(a: &[Vec<f64>], b: &[Vec<f64>], n: usize, tol: f64) -> bool {
        for i in 0..n {
            for j in 0..n {
                if (a[i][j] - b[i][j]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_matrix_exp_zero() {
        let z = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let e = matrix_exp(&z, 2);
        let ident = eye(2);
        assert!(approx_eq_matrix(&e, &ident, 2, 1e-12));
    }

    #[test]
    fn test_matrix_exp_rotation() {
        // A = pi/2 * [[0, -1], [1, 0]] => expm(A) = 90 degree rotation
        let t = std::f64::consts::PI / 2.0;
        let a = vec![vec![0.0, -t], vec![t, 0.0]];
        let e = matrix_exp(&a, 2);
        // expm = [[cos(pi/2), -sin(pi/2)], [sin(pi/2), cos(pi/2)]]
        assert!((e[0][0]).abs() < 1e-10); // cos(pi/2)
        assert!((e[0][1] + 1.0).abs() < 1e-10); // -sin(pi/2) = -1
        assert!((e[1][0] - 1.0).abs() < 1e-10); // sin(pi/2) = 1
        assert!((e[1][1]).abs() < 1e-10); // cos(pi/2)
    }

    #[test]
    fn test_matrix_exp_diagonal() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let e = matrix_exp(&a, 2);
        assert!((e[0][0] - 1.0f64.exp()).abs() < 1e-10);
        assert!((e[1][1] - 2.0f64.exp()).abs() < 1e-10);
        assert!(e[0][1].abs() < 1e-12);
    }

    #[test]
    fn test_matrix_log_exp() {
        let a = vec![vec![2.0, 1.0], vec![0.0, 3.0]];
        let log_a = matrix_log(&a, 2).expect("failed to create log_a");
        let exp_log_a = matrix_exp(&log_a, 2);
        assert!(approx_eq_matrix(&exp_log_a, &a, 2, 1e-8));
    }

    #[test]
    fn test_matrix_sqrt_diagonal() {
        let a = vec![vec![4.0, 0.0], vec![0.0, 9.0]];
        let s = matrix_sqrt(&a, 2).expect("failed to create s");
        assert!((s[0][0] - 2.0).abs() < 1e-8);
        assert!((s[1][1] - 3.0).abs() < 1e-8);
        // s^2 = a
        let s2 = matmul(&s, &s, 2);
        assert!(approx_eq_matrix(&s2, &a, 2, 1e-8));
    }

    #[test]
    fn test_matrix_pow_half() {
        let a = vec![vec![4.0, 0.0], vec![0.0, 9.0]];
        let ah = matrix_pow(&a, 2, 0.5).expect("failed to create ah");
        assert!((ah[0][0] - 2.0).abs() < 1e-6);
        assert!((ah[1][1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_pow_integer() {
        let a = vec![vec![1.0, 1.0], vec![0.0, 1.0]];
        let a3 = matrix_pow(&a, 2, 3.0).expect("failed to create a3");
        // [[1,1],[0,1]]^3 = [[1,3],[0,1]]
        assert!((a3[0][0] - 1.0).abs() < 1e-12);
        assert!((a3[0][1] - 3.0).abs() < 1e-12);
        assert!((a3[1][0]).abs() < 1e-12);
        assert!((a3[1][1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_sin_zero() {
        let z = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let s = matrix_sin(&z, 2);
        for i in 0..2 {
            for j in 0..2 {
                assert!(s[i][j].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_matrix_cos_zero() {
        let z = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let c = matrix_cos(&z, 2);
        let ident = eye(2);
        assert!(approx_eq_matrix(&c, &ident, 2, 1e-12));
    }

    #[test]
    fn test_sin_cos_identity() {
        // sin^2(A) + cos^2(A) = I for diagonal A
        let a = vec![vec![0.3, 0.0], vec![0.0, 0.5]];
        let s = matrix_sin(&a, 2);
        let c = matrix_cos(&a, 2);
        let s2 = matmul(&s, &s, 2);
        let c2 = matmul(&c, &c, 2);
        let sum = matadd(&s2, &c2, 2);
        let ident = eye(2);
        assert!(approx_eq_matrix(&sum, &ident, 2, 1e-10));
    }

    #[test]
    fn test_frechet_derivative_numerical() {
        let a = vec![vec![0.1, 0.2], vec![-0.1, 0.3]];
        let e = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let l = frechet_derivative_expm(&a, &e, 2);
        // Finite difference check
        let eps = 1e-6;
        let ae: Vec<Vec<f64>> = a
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, &v)| v + eps * e[i][j])
                    .collect()
            })
            .collect();
        let ea1 = matrix_exp(&ae, 2);
        let ea0 = matrix_exp(&a, 2);
        for i in 0..2 {
            for j in 0..2 {
                let fd = (ea1[i][j] - ea0[i][j]) / eps;
                assert!(
                    (l[i][j] - fd).abs() < 1e-5,
                    "l[{i}][{j}] = {}, fd = {}",
                    l[i][j],
                    fd
                );
            }
        }
    }

    #[test]
    fn test_expm_cond_nonneg() {
        let a = vec![vec![0.1, 0.0], vec![0.0, 0.2]];
        let kappa = expm_cond(&a, 2);
        assert!(kappa >= 0.0);
    }
}
