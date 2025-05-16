//! Matrix functions for sparse matrices

use crate::csr::CsrMatrix;
use crate::error::SparseResult;
use crate::linalg::interface::LinearOperator;
use num_traits::{Float, NumAssign};
use rand::Rng;
use std::fmt::Debug;
use std::iter::Sum;

/// Compute the action of the matrix exponential on a vector: y = exp(t*A) * v
///
/// This function computes y = exp(t*A) * v without explicitly forming exp(t*A),
/// using a Krylov subspace approximation.
///
/// # Arguments
///
/// * `a` - The linear operator A
/// * `v` - The vector to multiply
/// * `t` - The scalar parameter (usually time)
/// * `m` - The dimension of the Krylov subspace (default: 30)
/// * `tol` - Tolerance for the approximation (default: 1e-7)
///
/// # Returns
///
/// The result vector y = exp(t*A) * v
pub fn expm_multiply<F>(
    a: &dyn LinearOperator<F>,
    v: &[F],
    t: F,
    m: Option<usize>,
    tol: Option<F>,
) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(crate::error::SparseError::ValueError(
            "Matrix must be square for expm_multiply".to_string(),
        ));
    }
    if v.len() != cols {
        return Err(crate::error::SparseError::DimensionMismatch {
            expected: cols,
            found: v.len(),
        });
    }

    let n = rows;
    let m = m.unwrap_or(30.min(n - 1));
    let tol = tol.unwrap_or(F::from(1e-7).unwrap());

    // Special case: t = 0
    if t == F::zero() {
        return Ok(v.to_vec());
    }

    // Normalize the initial vector
    let v_norm = norm2(v);
    if v_norm == F::zero() {
        return Ok(vec![F::zero(); n]);
    }

    let v_normalized: Vec<F> = v.iter().map(|&vi| vi / v_norm).collect();

    // Build Krylov subspace using Arnoldi iteration
    let mut v = vec![v_normalized.clone()]; // Orthonormal basis
    let mut h = vec![vec![F::zero(); m]; m + 1]; // Upper Hessenberg matrix

    for j in 0..m {
        // Compute w = A * v[j]
        let mut w = a.matvec(&v[j])?;

        // Orthogonalize w against previous vectors
        for i in 0..=j {
            let h_ij = dot(&v[i], &w);
            h[i][j] = h_ij;
            // w = w - h_ij * v[i]
            for k in 0..n {
                w[k] -= h_ij * v[i][k];
            }
        }

        let h_next = norm2(&w);
        h[j + 1][j] = h_next;

        // Check for breakdown
        if h_next.abs() < tol * F::from(100).unwrap() {
            // Early termination - Krylov subspace is exhausted
            break;
        }

        // Normalize w and add to basis
        let w_normalized: Vec<F> = w.iter().map(|&wi| wi / h_next).collect();
        v.push(w_normalized);
    }

    // Extract the square Hessenberg matrix
    let actual_m = v.len() - 1;

    // Special case: if Krylov subspace has dimension 1
    if actual_m == 0 {
        // For 1x1 case, h[0][0] contains the eigenvalue
        let lambda = h[0][0];
        let exp_t_lambda = (t * lambda).exp();
        let mut y = vec![F::zero(); n];
        for j in 0..n {
            y[j] = v_norm * exp_t_lambda * v[0][j];
        }
        return Ok(y);
    }

    let mut h_square = vec![vec![F::zero(); actual_m]; actual_m];
    for i in 0..actual_m {
        for j in 0..actual_m {
            h_square[i][j] = h[i][j];
        }
    }

    // Compute exp(t*H) using scaling and squaring
    let exp_t_h = matrix_exponential_dense(&h_square, t)?;

    // Compute y = v_norm * V * exp(t*H) * e1
    let mut y = vec![F::zero(); n];
    for i in 0..actual_m {
        let coeff = v_norm * exp_t_h[i][0];
        for j in 0..n {
            y[j] += coeff * v[i][j];
        }
    }

    Ok(y)
}

/// Compute matrix exponential of a small dense matrix
fn matrix_exponential_dense<F>(h: &[Vec<F>], t: F) -> SparseResult<Vec<Vec<F>>>
where
    F: Float + NumAssign + 'static,
{
    let _n = h.len();

    // Scaling: find s such that ||t*H/2^s|| < 1
    let mut s = 0;
    let mut scale = F::one();
    let h_norm = matrix_norm_inf(h);
    let mut scaled_norm = (t * h_norm).abs();

    while scaled_norm > F::one() {
        s += 1;
        scale *= F::from(2).unwrap();
        scaled_norm /= F::from(2).unwrap();
    }

    let t_scaled = t / scale;

    // Compute Padé approximation of exp(t_scaled * H)
    let mut exp_h = pade_approximation(h, t_scaled, 6)?;

    // Squaring phase: exp(t*H) = (exp(t*H/2^s))^(2^s)
    for _ in 0..s {
        exp_h = matrix_multiply_dense(&exp_h, &exp_h)?;
    }

    Ok(exp_h)
}

/// Padé approximation for matrix exponential
fn pade_approximation<F>(a: &[Vec<F>], t: F, order: usize) -> SparseResult<Vec<Vec<F>>>
where
    F: Float + NumAssign + 'static,
{
    let n = a.len();

    // Compute powers of t*A
    let mut t_a = vec![vec![F::zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            t_a[i][j] = t * a[i][j];
        }
    }

    let mut powers = vec![identity_matrix(n)];
    powers.push(t_a.clone());

    for p in 2..=order {
        let prev = &powers[p - 1];
        let next = matrix_multiply_dense(&t_a, prev)?;
        powers.push(next);
    }

    // Padé coefficients for order 6
    let num_coeffs = [
        F::one(),
        F::from(0.5).unwrap(),
        F::from(3.0 / 26.0).unwrap(),
        F::from(1.0 / 312.0).unwrap(),
        F::from(1.0 / 11232.0).unwrap(),
        F::from(1.0 / 506880.0).unwrap(),
        F::from(1.0 / 18811680.0).unwrap(),
    ];

    let den_coeffs = [
        F::one(),
        F::from(-0.5).unwrap(),
        F::from(3.0 / 26.0).unwrap(),
        F::from(-1.0 / 312.0).unwrap(),
        F::from(1.0 / 11232.0).unwrap(),
        F::from(-1.0 / 506880.0).unwrap(),
        F::from(1.0 / 18811680.0).unwrap(),
    ];

    // Compute numerator and denominator
    let mut num = zero_matrix(n);
    let mut den = zero_matrix(n);

    for (i, coeff) in num_coeffs.iter().enumerate().take(order + 1) {
        add_scaled_matrix(&mut num, &powers[i], *coeff);
    }

    for (i, coeff) in den_coeffs.iter().enumerate().take(order + 1) {
        add_scaled_matrix(&mut den, &powers[i], *coeff);
    }

    // Solve den * exp_A = num for exp_A
    solve_matrix_equation(&den, &num)
}

/// Estimate the 1-norm of a sparse matrix using a randomized algorithm
///
/// This function estimates ||A||_1 without explicitly computing all columns of A.
/// It uses a randomized algorithm that typically requires only a few matrix-vector products.
///
/// # Arguments
///
/// * `a` - The sparse matrix
/// * `t` - Number of iterations (default: 2)
/// * `itmax` - Maximum number of iterations for the iterative algorithm (default: 5)
///
/// # Returns
///
/// An estimate of the 1-norm of the matrix
pub fn onenormest<F>(a: &CsrMatrix<F>, t: Option<usize>, itmax: Option<usize>) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + 'static + Debug,
{
    let n = a.cols();
    let t = t.unwrap_or(2);
    let itmax = itmax.unwrap_or(5);

    if n == 0 {
        return Ok(F::zero());
    }

    // Handle small matrices directly
    if n <= 4 {
        return exact_onenorm(a);
    }

    // Initialize with random vectors
    let mut rng = rand::rng();
    let mut x = vec![vec![F::zero(); n]; t];
    for j in 0..t {
        for i in 0..n {
            x[j][i] = if rng.random::<bool>() {
                F::one()
            } else {
                -F::one()
            };
        }
    }

    // Make sure the first column is the all-ones vector
    for i in 0..n {
        x[0][i] = F::one();
    }

    let mut est = F::zero();
    let mut est_old = F::zero();
    let mut ind_best = vec![0; n];
    let mut s = vec![false; n];

    for _ in 0..itmax {
        // Compute Y = A^T * X
        let mut y = vec![vec![F::zero(); n]; t];
        let a_t = a.transpose();
        for j in 0..t {
            let mut y_j = vec![F::zero(); n];
            for row in 0..a_t.rows() {
                let row_range = a_t.row_range(row);
                let row_indices = &a_t.indices[row_range.clone()];
                let row_data = &a_t.data[row_range];

                let mut sum = F::zero();
                for (col_idx, &col) in row_indices.iter().enumerate() {
                    sum += row_data[col_idx] * x[j][col];
                }
                y_j[row] = sum;
            }
            y[j] = y_j;
        }

        // Find the column of Y with maximum 1-norm
        let mut max_norm = F::zero();
        let mut max_col = 0;
        for j in 0..t {
            let norm = onenorm_vec(&y[j]);
            if norm > max_norm {
                max_norm = norm;
                max_col = j;
            }
        }

        est = max_norm;

        // Check convergence
        if est <= est_old {
            break;
        }
        est_old = est;

        // Find indices of maximum absolute values in y[max_col]
        let mut abs_y: Vec<(F, usize)> = y[max_col]
            .iter()
            .enumerate()
            .map(|(i, &y_val)| (y_val.abs(), i))
            .collect();
        abs_y.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Update ind_best with largest elements not in S
        let mut count = 0;
        for (_, idx) in abs_y {
            if !s[idx] {
                ind_best[count] = idx;
                s[idx] = true;
                count += 1;
                if count >= t {
                    break;
                }
            }
        }

        // If we couldn't find enough new indices, we're done
        if count < t {
            break;
        }

        // Form Z = sign(y[max_col])
        let z: Vec<F> = y[max_col]
            .iter()
            .map(|&y_val| {
                if y_val >= F::zero() {
                    F::one()
                } else {
                    -F::one()
                }
            })
            .collect();

        // Compute w = A * Z
        let mut w = vec![F::zero(); a.rows()];
        for row in 0..a.rows() {
            let row_range = a.row_range(row);
            let row_indices = &a.indices[row_range.clone()];
            let row_data = &a.data[row_range];

            let mut sum = F::zero();
            for (col_idx, &col) in row_indices.iter().enumerate() {
                sum += row_data[col_idx] * z[col];
            }
            w[row] = sum;
        }

        // Update x with unit vectors at positions ind_best
        for j in 0..t {
            for i in 0..n {
                x[j][i] = F::zero();
            }
            x[j][ind_best[j]] = F::one();
        }

        // Update estimate
        let w_norm = onenorm_vec(&w);
        if w_norm > est {
            est = w_norm;
        }
    }

    Ok(est)
}

// Helper functions

/// Compute the exact 1-norm of a matrix
fn exact_onenorm<F>(a: &CsrMatrix<F>) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + 'static + Debug,
{
    let n = a.cols();
    let mut max_norm = F::zero();

    for j in 0..n {
        let mut col_sum = F::zero();
        for i in 0..a.rows() {
            let val = a.get(i, j);
            if val != F::zero() {
                col_sum += val.abs();
            }
        }
        if col_sum > max_norm {
            max_norm = col_sum;
        }
    }

    Ok(max_norm)
}

/// Compute the 1-norm of a vector
fn onenorm_vec<F: Float + Sum>(x: &[F]) -> F {
    x.iter().map(|&xi| xi.abs()).sum()
}

/// Compute the dot product of two vectors
fn dot<F: Float + Sum>(x: &[F], y: &[F]) -> F {
    x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum()
}

/// Compute the 2-norm of a vector
fn norm2<F: Float + Sum>(x: &[F]) -> F {
    dot(x, x).sqrt()
}

/// Compute the infinity norm of a matrix
fn matrix_norm_inf<F: Float>(a: &[Vec<F>]) -> F {
    let mut max_norm = F::zero();
    for row in a {
        let row_sum: F = row.iter().map(|&x| x.abs()).fold(F::zero(), |a, b| a + b);
        if row_sum > max_norm {
            max_norm = row_sum;
        }
    }
    max_norm
}

/// Create an identity matrix
fn identity_matrix<F: Float>(n: usize) -> Vec<Vec<F>> {
    let mut identity = vec![vec![F::zero(); n]; n];
    for i in 0..n {
        identity[i][i] = F::one();
    }
    identity
}

/// Create a zero matrix
fn zero_matrix<F: Float>(n: usize) -> Vec<Vec<F>> {
    vec![vec![F::zero(); n]; n]
}

/// Add a scaled matrix to another: A += alpha * B
fn add_scaled_matrix<F: Float + NumAssign>(a: &mut [Vec<F>], b: &[Vec<F>], alpha: F) {
    let n = a.len();
    for i in 0..n {
        for j in 0..n {
            a[i][j] += alpha * b[i][j];
        }
    }
}

/// Multiply two dense matrices
fn matrix_multiply_dense<F: Float + NumAssign>(
    a: &[Vec<F>],
    b: &[Vec<F>],
) -> SparseResult<Vec<Vec<F>>> {
    let n = a.len();
    let mut c = vec![vec![F::zero(); n]; n];

    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    Ok(c)
}

/// Solve the matrix equation AX = B for X
fn solve_matrix_equation<F: Float + NumAssign>(
    a: &[Vec<F>],
    b: &[Vec<F>],
) -> SparseResult<Vec<Vec<F>>> {
    let n = a.len();

    // LU decomposition
    let mut l = vec![vec![F::zero(); n]; n];
    let mut u = a.to_vec();

    for i in 0..n {
        l[i][i] = F::one();
    }

    // Gaussian elimination
    if n > 1 {
        for k in 0..n - 1 {
            for i in k + 1..n {
                if u[k][k].abs() < F::epsilon() {
                    return Err(crate::error::SparseError::SingularMatrix(
                        "Matrix is singular".to_string(),
                    ));
                }
                let factor = u[i][k] / u[k][k];
                l[i][k] = factor;
                for j in k..n {
                    u[i][j] = u[i][j] - factor * u[k][j];
                }
            }
        }
    }

    // Solve LY = B for Y
    let mut y = vec![vec![F::zero(); n]; n];
    for j in 0..n {
        for i in 0..n {
            let mut sum = b[i][j];
            for k in 0..i {
                sum -= l[i][k] * y[k][j];
            }
            y[i][j] = sum;
        }
    }

    // Solve UX = Y for X
    let mut x = vec![vec![F::zero(); n]; n];
    for j in 0..n {
        for i in (0..n).rev() {
            let mut sum = y[i][j];
            for k in i + 1..n {
                sum -= u[i][k] * x[k][j];
            }
            if u[i][i].abs() < F::epsilon() {
                return Err(crate::error::SparseError::SingularMatrix(
                    "Matrix is singular".to_string(),
                ));
            }
            x[i][j] = sum / u[i][i];
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::interface::{IdentityOperator, ScaledIdentityOperator};

    #[test]
    fn test_expm_multiply_identity() {
        // exp(t*I) * v = exp(t) * v
        let identity = IdentityOperator::<f64>::new(3);
        let v = vec![1.0, 2.0, 3.0];
        let t = 1.0;

        let result = expm_multiply(&identity, &v, t, None, None).unwrap();

        let exp_t = t.exp();
        let expected: Vec<f64> = v.iter().map(|&vi| exp_t * vi).collect();

        println!("Identity test - result: {:?}", result);
        println!("Identity test - expected: {:?}", expected);
        println!("Identity test - exp(t): {}", exp_t);

        for (ri, ei) in result.iter().zip(&expected) {
            assert!((ri - ei).abs() < 1e-10);
        }
    }

    #[test]
    fn test_expm_multiply_scaled_identity() {
        // exp(t*alpha*I) * v = exp(t*alpha) * v
        let alpha = 2.0;
        let scaled_identity = ScaledIdentityOperator::new(3, alpha);
        let v = vec![1.0, 2.0, 3.0];
        let t = 0.5;

        let result = expm_multiply(&scaled_identity, &v, t, None, None).unwrap();

        let exp_t_alpha = (t * alpha).exp();
        let expected: Vec<f64> = v.iter().map(|&vi| exp_t_alpha * vi).collect();

        for (ri, ei) in result.iter().zip(&expected) {
            assert!((ri - ei).abs() < 1e-6);
        }
    }

    #[test]
    fn test_expm_multiply_zero_time() {
        // exp(0*A) * v = v
        let identity = IdentityOperator::<f64>::new(3);
        let v = vec![1.0, 2.0, 3.0];
        let t = 0.0;

        let result = expm_multiply(&identity, &v, t, None, None).unwrap();

        for (ri, vi) in result.iter().zip(&v) {
            assert!((ri - vi).abs() < 1e-10);
        }
    }

    #[test]
    fn test_onenormest_small_matrix() {
        // For small matrices, it should compute the exact norm
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![2.0, 3.0, 4.0];
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let estimate = onenormest(&matrix, None, None).unwrap();

        // For a diagonal matrix, the 1-norm is the maximum absolute diagonal element
        assert!((estimate - 4.0).abs() < 1e-10);
    }
}
