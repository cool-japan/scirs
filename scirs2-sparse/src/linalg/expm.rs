//! Matrix exponential computation for sparse matrices
//!
//! This module implements the matrix exponential using the scaling and squaring
//! method with Padé approximation.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use num_traits::{Float, NumAssign, One, Zero};
use std::iter::Sum;

/// Compute the matrix exponential using scaling and squaring with Padé approximation
///
/// This function computes exp(A) for a sparse matrix A using the scaling and
/// squaring method combined with Padé approximation.
///
/// # Arguments
///
/// * `a` - The sparse matrix A (must be square)
///
/// # Returns
///
/// The matrix exponential exp(A) as a sparse matrix
///
/// # Implementation Details
///
/// Uses 13th order Padé approximation for high accuracy (machine precision).
/// The algorithm automatically selects the appropriate scaling factor based
/// on the matrix norm to ensure numerical stability.
pub fn expm<F>(a: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Matrix must be square for expm".to_string(),
        ));
    }

    // Compute the matrix infinity norm
    let a_norm = matrix_inf_norm(a)?;

    // Constants for order 13 Padé approximation
    let theta_13 = F::from(5.371920351148152).unwrap();

    // If the norm is small enough, use direct Padé approximation
    if a_norm <= theta_13 {
        return pade_approximation(a, 13);
    }

    // Otherwise, use scaling and squaring
    // Find s such that ||A/2^s|| <= theta_13
    let mut s = 0;
    let mut scaled_norm = a_norm;
    let two = F::from(2.0).unwrap();

    while scaled_norm > theta_13 {
        s += 1;
        scaled_norm /= two;
    }

    // Compute A/2^s
    let scale_factor = two.powi(s);
    let scaled_a = scale_matrix(a, F::one() / scale_factor)?;

    // Compute exp(A/2^s) using Padé approximation
    let mut exp_scaled = pade_approximation(&scaled_a, 13)?;

    // Square the result s times to get exp(A)
    for _ in 0..s {
        exp_scaled = exp_scaled.matmul(&exp_scaled)?;
    }

    Ok(exp_scaled)
}

/// Compute the Padé approximation of exp(A)
///
/// Uses the diagonal Padé approximant of order (p,p)
fn pade_approximation<F>(a: &CsrMatrix<F>, p: usize) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    let n = a.shape().0;

    // Compute powers of A
    let mut a_powers = vec![sparse_identity(n)?]; // A^0 = I
    a_powers.push(a.clone()); // A^1 = A

    // Compute A^2, A^3, ..., A^p
    for i in 2..=p {
        let prev = &a_powers[i - 1];
        let power = prev.matmul(a)?;
        a_powers.push(power);
    }

    // Compute Padé coefficients
    let pade_coeffs = match p {
        6 => vec![
            F::from(1.0).unwrap(),
            F::from(1.0 / 2.0).unwrap(),
            F::from(3.0 / 26.0).unwrap(),
            F::from(1.0 / 312.0).unwrap(),
            F::from(1.0 / 10608.0).unwrap(),
            F::from(1.0 / 358800.0).unwrap(),
            F::from(1.0 / 17297280.0).unwrap(),
        ],
        13 => {
            // Compute coefficients for Padé (13,13) approximant
            // c_k = (2p-k)! p! / ((2p)! k! (p-k)!) for k = 0, 1, ..., p
            let two_p = 26i64;
            let p = 13i64;
            let mut coeffs = Vec::with_capacity(14);

            for k in 0..=p {
                let mut num = 1.0;
                let mut den = 1.0;

                // (2p-k)! / (2p)!
                for i in (two_p - k + 1)..=two_p {
                    den *= i as f64;
                }

                // p! / (p-k)!
                for i in (p - k + 1)..=p {
                    num *= i as f64;
                }

                // 1 / k!
                let mut k_fact = 1.0;
                for i in 1..=k {
                    k_fact *= i as f64;
                }

                coeffs.push(F::from(num / (den * k_fact)).unwrap());
            }

            coeffs
        }
        _ => {
            // General formula for Padé coefficients
            let mut coeffs = vec![F::zero(); p + 1];
            let mut factorial: F = F::one();
            for (i, coeff) in coeffs.iter_mut().enumerate().take(p + 1) {
                if i > 0 {
                    factorial *= F::from(i).unwrap();
                }
                let numerator = factorial;
                let mut denominator = F::one();
                for j in 1..=i {
                    denominator *= F::from(p + 1 - j).unwrap();
                }
                for j in 1..=(p - i) {
                    denominator *= F::from(j).unwrap();
                }
                *coeff = numerator / denominator;
            }
            coeffs
        }
    };

    // Compute U and V for the Padé approximant
    let mut u = sparse_zero(n)?;
    let mut v = sparse_zero(n)?;

    // U = sum of odd powers, V = sum of even powers
    for (i, coeff) in pade_coeffs.iter().enumerate() {
        let scaled_matrix = scale_matrix(&a_powers[i], *coeff)?;
        if i % 2 == 0 {
            v = sparse_add(&v, &scaled_matrix)?;
        } else {
            u = sparse_add(&u, &scaled_matrix)?;
        }
    }

    // Compute (V - U)^(-1) * (V + U)
    let neg_u = scale_matrix(&u, F::from(-1.0).unwrap())?;
    let v_minus_u = sparse_add(&v, &neg_u)?;
    let v_plus_u = sparse_add(&v, &u)?;

    // Solve (V - U) * X = (V + U) for X
    sparse_solve(&v_minus_u, &v_plus_u)
}

/// Compute the infinity norm of a sparse matrix
fn matrix_inf_norm<F>(a: &CsrMatrix<F>) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + std::fmt::Debug,
{
    let mut max_row_sum = F::zero();

    // For CSR format, efficiently compute row sums
    for row in 0..a.rows() {
        let start = a.indptr[row];
        let end = a.indptr[row + 1];
        let row_sum: F = a.data[start..end].iter().map(|x| x.abs()).sum();

        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }

    Ok(max_row_sum)
}

/// Scale a sparse matrix by a scalar
fn scale_matrix<F>(a: &CsrMatrix<F>, scale: F) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign,
{
    let mut data = a.data.clone();
    for val in data.iter_mut() {
        *val *= scale;
    }
    CsrMatrix::from_raw_csr(data, a.indptr.clone(), a.indices.clone(), a.shape())
}

/// Create a sparse identity matrix
fn sparse_identity<F>(n: usize) -> SparseResult<CsrMatrix<F>>
where
    F: Float + Zero + One,
{
    let mut rows = Vec::with_capacity(n);
    let mut cols = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    for i in 0..n {
        rows.push(i);
        cols.push(i);
        values.push(F::one());
    }

    CsrMatrix::new(values, rows, cols, (n, n))
}

/// Create a sparse zero matrix
fn sparse_zero<F>(n: usize) -> SparseResult<CsrMatrix<F>>
where
    F: Float + Zero,
{
    Ok(CsrMatrix::empty((n, n)))
}

/// Add two sparse matrices
fn sparse_add<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign,
{
    if a.shape() != b.shape() {
        return Err(SparseError::ShapeMismatch {
            expected: a.shape(),
            found: b.shape(),
        });
    }

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut values = Vec::new();

    for i in 0..a.rows() {
        for j in 0..a.cols() {
            let val = a.get(i, j) + b.get(i, j);
            if val.abs() > F::epsilon() {
                rows.push(i);
                cols.push(j);
                values.push(val);
            }
        }
    }

    CsrMatrix::new(values, rows, cols, a.shape())
}

/// Solve a linear system A * X = B for sparse matrices
///
/// Note: This is a placeholder - in practice you'd use a more sophisticated solver
fn sparse_solve<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    use crate::linalg::interface::MatrixLinearOperator;
    use crate::linalg::iterative::bicgstab;
    use crate::linalg::iterative::BiCGSTABOptions;

    let n = a.rows();
    let mut result_rows = Vec::new();
    let mut result_cols = Vec::new();
    let mut result_values = Vec::new();

    // Solve column by column
    for col in 0..b.cols() {
        // Extract the column from B
        let b_col = (0..n).map(|row| b.get(row, col)).collect::<Vec<_>>();

        // Create a linear operator for the matrix
        let op = MatrixLinearOperator::new(a.clone());

        // Create options for BiCGSTAB
        let options = BiCGSTABOptions {
            rtol: F::from(1e-10).unwrap(),
            atol: F::from(1e-12).unwrap(),
            max_iter: 1000,
            x0: None,
            left_preconditioner: None,
            right_preconditioner: None,
        };

        // Use BiCGSTAB to solve A * x = b_col
        let result = bicgstab(&op, &b_col, options)?;

        // Check convergence
        if !result.converged {
            return Err(SparseError::IterativeSolverFailure(format!(
                "BiCGSTAB failed to converge in {} iterations",
                result.iterations
            )));
        }

        // Add non-zero entries to result
        for (row, &val) in result.x.iter().enumerate() {
            if val.abs() > F::epsilon() {
                result_rows.push(row);
                result_cols.push(col);
                result_values.push(val);
            }
        }
    }

    CsrMatrix::new(result_values, result_rows, result_cols, (n, b.cols()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_expm_identity() {
        // exp(0) = I
        let n = 3;
        let zero_matrix = sparse_zero::<f64>(n).unwrap();
        let exp_zero = expm(&zero_matrix).unwrap();

        // Check that exp(0) is identity
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = exp_zero.get(i, j);
                assert_relative_eq!(actual, expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_expm_diagonal() {
        // For diagonal matrix D, exp(D) is diagonal with exp(d_ii) on diagonal
        let n = 3;
        let diag_values = vec![0.5, 1.0, 2.0];
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();

        for (i, &val) in diag_values.iter().enumerate() {
            rows.push(i);
            cols.push(i);
            values.push(val);
        }

        let diag_matrix = CsrMatrix::new(values, rows, cols, (n, n)).unwrap();
        let exp_diag = expm(&diag_matrix).unwrap();

        // Check diagonal values with high precision
        for (i, &val) in diag_values.iter().enumerate() {
            let expected = val.exp();
            let actual = exp_diag.get(i, i);
            assert_relative_eq!(actual, expected, epsilon = 1e-10);
        }

        // Check off-diagonal values are zero
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let actual = exp_diag.get(i, j);
                    assert_relative_eq!(actual, 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_expm_small_matrix() {
        // Test on a small 2x2 matrix with known exponential
        // A = [[0, 1], [0, 0]]
        // exp(A) = [[1, 1], [0, 1]]
        let rows = vec![0, 1];
        let cols = vec![1, 0];
        let values = vec![1.0, 0.0];

        let a = CsrMatrix::new(values, rows, cols, (2, 2)).unwrap();
        let exp_a = expm(&a).unwrap();

        // Check the result
        assert_relative_eq!(exp_a.get(0, 0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(exp_a.get(0, 1), 1.0, epsilon = 1e-10);
        assert_relative_eq!(exp_a.get(1, 0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(exp_a.get(1, 1), 1.0, epsilon = 1e-10);
    }
}
