//! Matrix functions for sparse matrices

use crate::csr::CsrMatrix;
use crate::error::SparseResult;
use crate::linalg::interface::LinearOperator;
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::Rng;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};

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
#[allow(dead_code)]
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
            for (k, w_val) in w.iter_mut().enumerate().take(n) {
                *w_val -= h_ij * v[i][k];
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
        for (j, y_val) in y.iter_mut().enumerate().take(n) {
            *y_val = v_norm * exp_t_lambda * v[0][j];
        }
        return Ok(y);
    }

    let mut h_square = vec![vec![F::zero(); actual_m]; actual_m];
    for (i, h_row) in h.iter().enumerate().take(actual_m) {
        for (j, &h_val) in h_row.iter().enumerate().take(actual_m) {
            h_square[i][j] = h_val;
        }
    }

    // Compute exp(t*H) using scaling and squaring
    let exp_t_h = matrix_exponential_dense(&h_square, t)?;

    // Compute y = v_norm * V * exp(t*H) * e1
    let mut y = vec![F::zero(); n];
    for (i, v_row) in v.iter().enumerate().take(actual_m) {
        let coeff = v_norm * exp_t_h[i][0];
        for (j, y_val) in y.iter_mut().enumerate().take(n) {
            *y_val += coeff * v_row[j];
        }
    }

    Ok(y)
}

/// Compute matrix exponential of a small dense matrix
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Estimate the 2-norm (spectral norm) of a sparse matrix using power iteration
///
/// This function estimates ||A||_2 = σ_max(A), the largest singular value of A,
/// using power iteration on A^T * A. The 2-norm is also known as the spectral norm.
///
/// # Arguments
///
/// * `a` - The sparse matrix
/// * `tol` - Convergence tolerance (default: 1e-6)
/// * `maxiter` - Maximum number of power iterations (default: 100)
///
/// # Returns
///
/// An estimate of the 2-norm of the matrix
///
/// # Examples
///
/// ```no_run
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::linalg::twonormest;
///
/// let rows = vec![0, 1];
/// let cols = vec![0, 1];
/// let data = vec![2.0, 3.0];
/// let matrix = CsrMatrix::new(data, rows, cols, (2, 2)).unwrap();
///
/// let norm_estimate = twonormest(&matrix, None, Some(1)).unwrap();
/// ```
#[allow(dead_code)]
pub fn twonormest<F>(a: &CsrMatrix<F>, tol: Option<F>, maxiter: Option<usize>) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + 'static + Debug,
{
    let (m, n) = (a.rows(), a.cols());
    let tol = tol.unwrap_or_else(|| F::from(1e-6).unwrap());
    let maxiter = maxiter.unwrap_or(100);

    if m == 0 || n == 0 {
        return Ok(F::zero());
    }

    // For very small matrices, compute exactly
    if n <= 4 && m <= 4 {
        return exact_twonorm(a);
    }

    // Initialize with a random unit vector
    let mut rng = scirs2_core::random::rng();
    let mut v: Vec<F> = (0..n)
        .map(|_| F::from(rng.random::<f64>() - 0.5).unwrap())
        .collect();

    // Normalize initial vector
    let v_norm = norm2(&v);
    if v_norm == F::zero() {
        return Ok(F::zero());
    }
    for v_elem in v.iter_mut() {
        *v_elem /= v_norm;
    }

    let mut lambda = F::zero();
    let mut lambda_old = F::zero();

    for iter in 0..maxiter {
        // Compute w = A * v
        let mut w = vec![F::zero(); m];
        for (row, w_val) in w.iter_mut().enumerate().take(m) {
            let row_range = a.row_range(row);
            let row_indices = &a.indices[row_range.clone()];
            let row_data = &a.data[row_range];

            let mut sum = F::zero();
            for (col_idx, &col) in row_indices.iter().enumerate() {
                sum += row_data[col_idx] * v[col];
            }
            *w_val = sum;
        }

        // Compute u = A^T * w
        let mut u = vec![F::zero(); n];
        let a_t = a.transpose();
        for (row, u_val) in u.iter_mut().enumerate().take(a_t.rows()) {
            let row_range = a_t.row_range(row);
            let row_indices = &a_t.indices[row_range.clone()];
            let row_data = &a_t.data[row_range];

            let mut sum = F::zero();
            for (col_idx, &col) in row_indices.iter().enumerate() {
                sum += row_data[col_idx] * w[col];
            }
            *u_val = sum;
        }

        // Estimate eigenvalue: λ = v^T * u
        lambda = dot(&v, &u);

        // Normalize u to get new v
        let u_norm = norm2(&u);
        if u_norm == F::zero() {
            break;
        }

        for (i, u_val) in u.iter().enumerate() {
            v[i] = *u_val / u_norm;
        }

        // Check convergence
        if iter > 0 {
            let rel_change = if lambda_old != F::zero() {
                ((lambda - lambda_old) / lambda_old).abs()
            } else {
                lambda.abs()
            };

            if rel_change < tol {
                break;
            }
        }

        lambda_old = lambda;
    }

    // The 2-norm is the square root of the largest eigenvalue of A^T * A
    Ok(lambda.sqrt())
}

/// Estimate the condition number of a sparse matrix
///
/// This function estimates cond(A) = ||A||_2 * ||A^(-1)||_2 using norm estimation.
/// For efficiency, it estimates ||A^(-1)||_2 by solving (A^T * A) * x = b for random b
/// and using power iteration to estimate the smallest singular value.
///
/// # Arguments
///
/// * `a` - The sparse matrix
/// * `norm_type` - The norm to use: "1" for 1-norm, "2" for 2-norm (default: "2")
/// * `tol` - Convergence tolerance (default: 1e-6)
/// * `maxiter` - Maximum number of iterations (default: 100)
///
/// # Returns
///
/// An estimate of the condition number
///
/// # Examples
///
/// ```no_run
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::linalg::condest;
///
/// let rows = vec![0, 1];
/// let cols = vec![0, 1];
/// let data = vec![2.0, 3.0];
/// let matrix = CsrMatrix::new(data, rows, cols, (2, 2)).unwrap();
///
/// let cond_estimate = condest(&matrix, None, None, Some(1)).unwrap();
/// ```
#[allow(dead_code)]
pub fn condest<F>(
    a: &CsrMatrix<F>,
    norm_type: Option<&str>,
    tol: Option<F>,
    maxiter: Option<usize>,
) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + 'static + Debug,
{
    let (m, n) = (a.rows(), a.cols());
    if m != n {
        return Err(crate::error::SparseError::ValueError(
            "Condition number estimation requires a square matrix".to_string(),
        ));
    }

    let norm_type = norm_type.unwrap_or("2");
    let tol = tol.unwrap_or_else(|| F::from(1e-6).unwrap());
    let maxiter = maxiter.unwrap_or(100);

    // Estimate ||A||
    let norm_a = match norm_type {
        "1" => onenormest(a, None, None)?,
        "2" => twonormest(a, Some(tol), Some(maxiter))?,
        _ => {
            return Err(crate::error::SparseError::ValueError(
                "norm_type must be '1' or '2'".to_string(),
            ))
        }
    };

    if norm_a == F::zero() {
        return Ok(F::infinity());
    }

    // Estimate ||A^(-1)|| using inverse power iteration
    // This estimates the smallest singular value σ_min, then ||A^(-1)||_2 = 1/σ_min
    let norm_a_inv = estimate_inverse_norm(a, norm_type, tol, maxiter)?;

    Ok(norm_a * norm_a_inv)
}

/// Estimate ||A^(-1)|| using inverse power iteration
#[allow(dead_code)]
fn estimate_inverse_norm<F>(
    a: &CsrMatrix<F>,
    norm_type: &str,
    tol: F,
    maxiter: usize,
) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + 'static + Debug,
{
    let _n = a.rows();

    // For 1-norm, use inverse power iteration with A^(-1)
    if norm_type == "1" {
        // This is more complex and would require solving linear systems
        // For now, we'll use a simpler 2-norm based approach
        return estimate_smallest_singular_value(a, tol, maxiter).map(|sigma_min| {
            if sigma_min == F::zero() {
                F::infinity()
            } else {
                F::one() / sigma_min
            }
        });
    }

    // For 2-norm, estimate smallest singular value of A
    estimate_smallest_singular_value(a, tol, maxiter).map(|sigma_min| {
        if sigma_min == F::zero() {
            F::infinity()
        } else {
            F::one() / sigma_min
        }
    })
}

/// Estimate the smallest singular value using inverse power iteration on A^T * A
#[allow(dead_code)]
fn estimate_smallest_singular_value<F>(a: &CsrMatrix<F>, tol: F, maxiter: usize) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + 'static + Debug,
{
    let n = a.cols();

    // Initialize with a random unit vector
    let mut rng = scirs2_core::random::rng();
    let mut v: Vec<F> = (0..n)
        .map(|_| F::from(rng.random::<f64>() - 0.5).unwrap())
        .collect();

    // Normalize initial vector
    let v_norm = norm2(&v);
    if v_norm == F::zero() {
        return Ok(F::zero());
    }
    for v_elem in v.iter_mut() {
        *v_elem /= v_norm;
    }

    let mut lambda = F::zero();
    let mut lambda_old = F::infinity();

    for iter in 0..maxiter {
        // We want to find the smallest eigenvalue of A^T * A
        // Using inverse iteration: solve (A^T * A) * x = v for x
        // For simplicity, we'll use a few iterations of the power method on (A^T * A)^(-1)
        // This is equivalent to solving (A^T * A) * u = v and then normalizing u

        // Since we don't have a direct solver, we'll approximate using several steps
        // of a simple iterative method (like Jacobi or minimal residual)
        let u = solve_ata_approximately(a, &v, 5)?; // 5 inner iterations

        // Normalize u
        let u_norm = norm2(&u);
        if u_norm == F::zero() {
            break;
        }

        for (i, u_val) in u.iter().enumerate() {
            v[i] = *u_val / u_norm;
        }

        // Estimate eigenvalue: λ = v^T * (A^T * A) * v
        lambda = estimate_rayleigh_quotient(a, &v)?;

        // Check convergence
        if iter > 0 {
            let rel_change = if lambda != F::zero() {
                ((lambda - lambda_old) / lambda).abs()
            } else {
                F::infinity()
            };

            if rel_change < tol {
                break;
            }
        }

        lambda_old = lambda;
    }

    // The smallest singular value is the square root of the smallest eigenvalue of A^T * A
    Ok(lambda.sqrt())
}

/// Approximately solve (A^T * A) * x = b using simple iteration
#[allow(dead_code)]
fn solve_ata_approximately<F>(
    a: &CsrMatrix<F>,
    b: &[F],
    num_iterations: usize,
) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + 'static + Debug,
{
    let n = a.cols();
    let mut x = b.to_vec(); // Initial guess

    for _ in 0..num_iterations {
        // Compute r = A^T * A * x
        let mut ax = vec![F::zero(); a.rows()];
        for (row, ax_val) in ax.iter_mut().enumerate().take(a.rows()) {
            let row_range = a.row_range(row);
            let row_indices = &a.indices[row_range.clone()];
            let row_data = &a.data[row_range];

            let mut sum = F::zero();
            for (col_idx, &col) in row_indices.iter().enumerate() {
                sum += row_data[col_idx] * x[col];
            }
            *ax_val = sum;
        }

        let mut ata_x = vec![F::zero(); n];
        let a_t = a.transpose();
        for (row, ata_x_val) in ata_x.iter_mut().enumerate().take(a_t.rows()) {
            let row_range = a_t.row_range(row);
            let row_indices = &a_t.indices[row_range.clone()];
            let row_data = &a_t.data[row_range];

            let mut sum = F::zero();
            for (col_idx, &col) in row_indices.iter().enumerate() {
                sum += row_data[col_idx] * ax[col];
            }
            *ata_x_val = sum;
        }

        // Simple iteration: x = x - α * (A^T * A * x - b)
        let alpha = F::from(0.1).unwrap(); // Simple step size
        for i in 0..n {
            x[i] -= alpha * (ata_x[i] - b[i]);
        }
    }

    Ok(x)
}

/// Estimate the Rayleigh quotient v^T * (A^T * A) * v
#[allow(dead_code)]
fn estimate_rayleigh_quotient<F>(a: &CsrMatrix<F>, v: &[F]) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + 'static + Debug,
{
    // Compute A * v
    let mut av = vec![F::zero(); a.rows()];
    for (row, av_val) in av.iter_mut().enumerate().take(a.rows()) {
        let row_range = a.row_range(row);
        let row_indices = &a.indices[row_range.clone()];
        let row_data = &a.data[row_range];

        let mut sum = F::zero();
        for (col_idx, &col) in row_indices.iter().enumerate() {
            sum += row_data[col_idx] * v[col];
        }
        *av_val = sum;
    }

    // Compute A^T * (A * v)
    let mut ata_v = vec![F::zero(); a.cols()];
    let a_t = a.transpose();
    for (row, ata_v_val) in ata_v.iter_mut().enumerate().take(a_t.rows()) {
        let row_range = a_t.row_range(row);
        let row_indices = &a_t.indices[row_range.clone()];
        let row_data = &a_t.data[row_range];

        let mut sum = F::zero();
        for (col_idx, &col) in row_indices.iter().enumerate() {
            sum += row_data[col_idx] * av[col];
        }
        *ata_v_val = sum;
    }

    // Return v^T * (A^T * A * v)
    Ok(dot(v, &ata_v))
}

/// Compute the exact 2-norm for small matrices
#[allow(dead_code)]
fn exact_twonorm<F>(a: &CsrMatrix<F>) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + 'static + Debug,
{
    // For small matrices, we can afford to compute all singular values
    // This is a simplified implementation - in practice, you'd use SVD

    let (m, n) = (a.rows(), a.cols());
    let min_dim = m.min(n);

    if min_dim == 0 {
        return Ok(F::zero());
    }

    if min_dim == 1 {
        // For 1D case, just compute the norm of the single row/column
        let mut max_norm = F::zero();
        for i in 0..m {
            for j in 0..n {
                let val = a.get(i, j).abs();
                if val > max_norm {
                    max_norm = val;
                }
            }
        }
        return Ok(max_norm);
    }

    // For small square matrices, use power iteration with high accuracy
    twonormest(a, Some(F::from(1e-12).unwrap()), Some(1000))
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
#[allow(dead_code)]
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
    let mut rng = scirs2_core::random::rng();
    let mut x = vec![vec![F::zero(); n]; t];
    for x_j in x.iter_mut().take(t) {
        for x_elem in x_j.iter_mut().take(n) {
            *x_elem = if rng.random::<bool>() {
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
            for (row, y_val) in y_j.iter_mut().enumerate().take(a_t.rows()) {
                let row_range = a_t.row_range(row);
                let row_indices = &a_t.indices[row_range.clone()];
                let row_data = &a_t.data[row_range];

                let mut sum = F::zero();
                for (col_idx, &col) in row_indices.iter().enumerate() {
                    sum += row_data[col_idx] * x[j][col];
                }
                *y_val = sum;
            }
            y[j] = y_j;
        }

        // Find the column of Y with maximum 1-norm
        let mut max_norm = F::zero();
        let mut max_col = 0;
        for (j, y_vec) in y.iter().enumerate().take(t) {
            let norm = onenorm_vec(y_vec);
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
        for (row, w_val) in w.iter_mut().enumerate().take(a.rows()) {
            let row_range = a.row_range(row);
            let row_indices = &a.indices[row_range.clone()];
            let row_data = &a.data[row_range];

            let mut sum = F::zero();
            for (col_idx, &col) in row_indices.iter().enumerate() {
                sum += row_data[col_idx] * z[col];
            }
            *w_val = sum;
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
#[allow(dead_code)]
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
#[allow(dead_code)]
fn onenorm_vec<F: Float + Sum>(x: &[F]) -> F {
    x.iter().map(|&xi| xi.abs()).sum()
}

/// Compute the dot product of two vectors
#[allow(dead_code)]
fn dot<F: Float + Sum>(x: &[F], y: &[F]) -> F {
    x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum()
}

/// Compute the 2-norm of a vector
#[allow(dead_code)]
fn norm2<F: Float + Sum>(x: &[F]) -> F {
    dot(x, x).sqrt()
}

/// Compute the infinity norm of a matrix
#[allow(dead_code)]
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
#[allow(dead_code)]
fn identity_matrix<F: Float>(n: usize) -> Vec<Vec<F>> {
    let mut identity = vec![vec![F::zero(); n]; n];
    for (i, row) in identity.iter_mut().enumerate().take(n) {
        row[i] = F::one();
    }
    identity
}

/// Create a zero matrix
#[allow(dead_code)]
fn zero_matrix<F: Float>(n: usize) -> Vec<Vec<F>> {
    vec![vec![F::zero(); n]; n]
}

/// Add a scaled matrix to another: A += alpha * B
#[allow(dead_code)]
fn add_scaled_matrix<F: Float + NumAssign>(a: &mut [Vec<F>], b: &[Vec<F>], alpha: F) {
    let n = a.len();
    for i in 0..n {
        for j in 0..n {
            a[i][j] += alpha * b[i][j];
        }
    }
}

/// Multiply two dense matrices
#[allow(dead_code)]
fn matrix_multiply_dense<F: Float + NumAssign>(
    a: &[Vec<F>],
    b: &[Vec<F>],
) -> SparseResult<Vec<Vec<F>>> {
    let n = a.len();
    let mut c = vec![vec![F::zero(); n]; n];

    for (i, c_row) in c.iter_mut().enumerate().take(n) {
        for (j, c_val) in c_row.iter_mut().enumerate().take(n) {
            for (k, &a_val) in a[i].iter().enumerate().take(n) {
                *c_val += a_val * b[k][j];
            }
        }
    }

    Ok(c)
}

/// Solve the matrix equation AX = B for X
#[allow(dead_code)]
fn solve_matrix_equation<F: Float + NumAssign>(
    a: &[Vec<F>],
    b: &[Vec<F>],
) -> SparseResult<Vec<Vec<F>>> {
    let n = a.len();

    // LU decomposition
    let mut l = vec![vec![F::zero(); n]; n];
    let mut u = a.to_vec();

    for (i, l_row) in l.iter_mut().enumerate().take(n) {
        l_row[i] = F::one();
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
            for (k, &l_val) in l[i].iter().enumerate().take(i) {
                sum -= l_val * y[k][j];
            }
            y[i][j] = sum;
        }
    }

    // Solve UX = Y for X
    let mut x = vec![vec![F::zero(); n]; n];
    for j in 0..n {
        for i in (0..n).rev() {
            let mut sum = y[i][j];
            for (k, &u_val) in u[i].iter().enumerate().skip(i + 1).take(n - i - 1) {
                sum -= u_val * x[k][j];
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

/// Enhanced 2-norm estimation for sparse arrays using power iteration
///
/// This function estimates ||A||_2 = σ_max(A), the largest singular value of A,
/// using power iteration on A^T * A. This version works with the newer CsrArray
/// and SparseArray types and includes improved convergence criteria.
///
/// # Arguments
///
/// * `a` - The sparse array (must implement SparseArray trait)
/// * `tol` - Convergence tolerance (default: 1e-8)
/// * `maxiter` - Maximum number of power iterations (default: 100)
/// * `initial_guess` - Optional initial guess vector
///
/// # Returns
///
/// An estimate of the 2-norm of the matrix
///
/// # Examples
///
/// ```no_run
/// use scirs2_sparse::csr_array::CsrArray;
/// use scirs2_sparse::linalg::twonormest_enhanced;
///
/// let rows = vec![0, 1];
/// let cols = vec![0, 1];
/// let data = vec![2.0, 3.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (2, 2), false).unwrap();
///
/// let norm_estimate = twonormest_enhanced(&matrix, None, Some(1), None).unwrap();
/// ```
#[allow(dead_code)]
pub fn twonormest_enhanced<T, S>(
    a: &S,
    tol: Option<T>,
    maxiter: Option<usize>,
    initial_guess: Option<ArrayView1<T>>,
) -> SparseResult<T>
where
    T: Float
        + NumAssign
        + Sum
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + scirs2_core::simd_ops::SimdUnifiedOps
        + Send
        + Sync,
    S: SparseArray<T>,
{
    let (m, n) = a.shape();
    let tol = tol.unwrap_or_else(|| T::from(1e-8).unwrap());
    let maxiter = maxiter.unwrap_or(100);

    if m == 0 || n == 0 {
        return Ok(T::zero());
    }

    // For very small matrices, use more accurate computation
    if n <= 4 && m <= 4 {
        return exact_twonorm_enhanced(a);
    }

    // Initialize with provided _guess or random unit vector
    let mut v = match initial_guess {
        Some(_guess) => {
            if _guess.len() != n {
                return Err(crate::error::SparseError::DimensionMismatch {
                    expected: n,
                    found: _guess.len(),
                });
            }
            _guess.to_owned()
        }
        None => {
            let mut rng = scirs2_core::random::rng();
            let mut v_arr = Array1::zeros(n);
            for i in 0..n {
                v_arr[i] = T::from(rng.random::<f64>() - 0.5).unwrap();
            }
            v_arr
        }
    };

    // Normalize initial vector
    let v_norm = (v.iter().map(|&x| x * x).sum::<T>()).sqrt();
    if v_norm == T::zero() {
        return Ok(T::zero());
    }
    for i in 0..n {
        v[i] /= v_norm;
    }

    let mut lambda = T::zero();
    let mut lambda_old = T::zero();
    let mut _converged = false;

    for iter in 0..maxiter {
        // Compute w = A * v using sparse matrix-vector product
        let w = sparse_matvec(a, &v.view())?;

        // Compute u = A^T * w using sparse matrix-vector product with transpose
        let u = sparse_matvec_transpose(a, &w.view())?;

        // Estimate eigenvalue: λ = v^T * u (Rayleigh quotient for A^T * A)
        lambda = v.iter().zip(u.iter()).map(|(&vi, &ui)| vi * ui).sum();

        // Normalize u to get new v
        let u_norm = (u.iter().map(|&x| x * x).sum::<T>()).sqrt();
        if u_norm == T::zero() {
            break;
        }

        for i in 0..n {
            v[i] = u[i] / u_norm;
        }

        // Check convergence using relative change
        if iter > 0 {
            let rel_change = if lambda_old != T::zero() {
                ((lambda - lambda_old) / lambda_old).abs()
            } else {
                lambda.abs()
            };

            if rel_change < tol {
                _converged = true;
                break;
            }
        }

        lambda_old = lambda;
    }

    // The 2-norm is the square root of the largest eigenvalue of A^T * A
    Ok(lambda.sqrt())
}

/// Enhanced condition number estimation for sparse arrays
///
/// This function estimates cond(A) = ||A||_2 * ||A^(-1)||_2 using enhanced norm estimation.
/// It provides better accuracy and works with the newer SparseArray trait.
///
/// # Arguments
///
/// * `a` - The sparse array (must be square)
/// * `norm_type` - The norm to use: "1" for 1-norm, "2" for 2-norm (default: "2")
/// * `tol` - Convergence tolerance (default: 1e-8)
/// * `maxiter` - Maximum number of iterations (default: 100)
///
/// # Returns
///
/// An estimate of the condition number
///
/// # Examples
///
/// ```no_run
/// use scirs2_sparse::csr_array::CsrArray;
/// use scirs2_sparse::linalg::condest_enhanced;
///
/// let rows = vec![0, 1];
/// let cols = vec![0, 1];
/// let data = vec![2.0, 3.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (2, 2), false).unwrap();
///
/// let cond_estimate = condest_enhanced(&matrix, None, None, Some(1)).unwrap();
/// ```
#[allow(dead_code)]
pub fn condest_enhanced<T, S>(
    a: &S,
    norm_type: Option<&str>,
    tol: Option<T>,
    maxiter: Option<usize>,
) -> SparseResult<T>
where
    T: Float
        + NumAssign
        + Sum
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + scirs2_core::simd_ops::SimdUnifiedOps
        + Send
        + Sync,
    S: SparseArray<T>,
{
    let (m, n) = a.shape();
    if m != n {
        return Err(crate::error::SparseError::ValueError(
            "Condition number estimation requires a square matrix".to_string(),
        ));
    }

    let norm_type = norm_type.unwrap_or("2");
    let tol = tol.unwrap_or_else(|| T::from(1e-8).unwrap());
    let maxiter = maxiter.unwrap_or(100);

    // Estimate ||A||
    let norm_a = match norm_type {
        "2" => twonormest_enhanced(a, Some(tol), Some(maxiter), None)?,
        "1" => onenormest_enhanced(a, None, None)?,
        _ => {
            return Err(crate::error::SparseError::ValueError(
                "norm_type must be '1' or '2'".to_string(),
            ))
        }
    };

    if norm_a == T::zero() {
        return Ok(T::infinity());
    }

    // Estimate ||A^(-1)|| using inverse power iteration
    let norm_a_inv = estimate_inverse_norm_enhanced(a, norm_type, tol, maxiter)?;

    Ok(norm_a * norm_a_inv)
}

/// Enhanced 1-norm estimation for sparse arrays
///
/// This function estimates ||A||_1 using a randomized algorithm optimized for sparse matrices.
/// It works with the newer SparseArray trait and includes performance improvements.
///
/// # Arguments
///
/// * `a` - The sparse array
/// * `t` - Number of random vectors to use (default: 2)
/// * `itmax` - Maximum number of iterations (default: 5)
///
/// # Returns
///
/// An estimate of the 1-norm of the matrix
#[allow(dead_code)]
pub fn onenormest_enhanced<T, S>(a: &S, t: Option<usize>, itmax: Option<usize>) -> SparseResult<T>
where
    T: Float
        + NumAssign
        + Sum
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static,
    S: SparseArray<T>,
{
    let (_m, n) = a.shape();
    let t = t.unwrap_or(2);
    let itmax = itmax.unwrap_or(5);

    if n == 0 {
        return Ok(T::zero());
    }

    // Handle small matrices exactly
    if n <= 4 {
        return exact_onenorm_enhanced(a);
    }

    // Initialize with random ±1 vectors
    let mut rng = scirs2_core::random::rng();
    let mut x_vectors = Vec::with_capacity(t);

    for _ in 0..t {
        let mut x = Array1::zeros(n);
        for i in 0..n {
            x[i] = if rng.random::<bool>() {
                T::one()
            } else {
                -T::one()
            };
        }
        x_vectors.push(x);
    }

    // First vector is the all-ones vector for better convergence
    if !x_vectors.is_empty() {
        for i in 0..n {
            x_vectors[0][i] = T::one();
        }
    }

    let mut est = T::zero();
    let mut est_old = T::zero();

    for _iter in 0..itmax {
        // Compute Y = A^T * X
        let mut y_vectors = Vec::with_capacity(t);
        for x in &x_vectors {
            let y = sparse_matvec_transpose(a, &x.view())?;
            y_vectors.push(y);
        }

        // Find the column of Y with maximum 1-norm
        let mut max_norm = T::zero();
        let mut max_col = 0;
        for (j, y) in y_vectors.iter().enumerate() {
            let norm = y.iter().map(|&val| val.abs()).sum();
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

        // Update X based on the signs of Y[max_col]
        x_vectors.clear();
        let mut x = Array1::zeros(n);
        for i in 0..n {
            x[i] = if y_vectors[max_col][i] >= T::zero() {
                T::one()
            } else {
                -T::one()
            };
        }
        x_vectors.push(x);

        // Add additional random vectors if needed
        for _ in 1..t {
            let mut x = Array1::zeros(n);
            for i in 0..n {
                x[i] = if rng.random::<bool>() {
                    T::one()
                } else {
                    -T::one()
                };
            }
            x_vectors.push(x);
        }
    }

    Ok(est)
}

/// Estimate ||A^(-1)|| using enhanced inverse power iteration
#[allow(dead_code)]
fn estimate_inverse_norm_enhanced<T, S>(
    a: &S,
    norm_type: &str,
    tol: T,
    maxiter: usize,
) -> SparseResult<T>
where
    T: Float
        + NumAssign
        + Sum
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + scirs2_core::simd_ops::SimdUnifiedOps
        + Send
        + Sync,
    S: SparseArray<T>,
{
    // For both 1-norm and 2-norm, estimate smallest singular value of A
    let sigma_min = estimate_smallest_singular_value_enhanced(a, tol, maxiter)?;

    if sigma_min == T::zero() {
        Ok(T::infinity())
    } else {
        Ok(T::one() / sigma_min)
    }
}

/// Estimate the smallest singular value using enhanced inverse power iteration
#[allow(dead_code)]
fn estimate_smallest_singular_value_enhanced<T, S>(a: &S, tol: T, maxiter: usize) -> SparseResult<T>
where
    T: Float
        + NumAssign
        + Sum
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + scirs2_core::simd_ops::SimdUnifiedOps
        + Send
        + Sync,
    S: SparseArray<T>,
{
    let (_, n) = a.shape();

    // Initialize with a random unit vector
    let mut rng = scirs2_core::random::rng();
    let mut v = Array1::zeros(n);
    for i in 0..n {
        v[i] = T::from(rng.random::<f64>() - 0.5).unwrap();
    }

    // Normalize initial vector
    let v_norm = (v.iter().map(|&x| x * x).sum::<T>()).sqrt();
    if v_norm == T::zero() {
        return Ok(T::zero());
    }
    for i in 0..n {
        v[i] /= v_norm;
    }

    let mut lambda = T::zero();
    let mut lambda_old = T::infinity();

    for iter in 0..maxiter {
        // We want to find the smallest eigenvalue of A^T * A
        // Using inverse iteration: solve (A^T * A) * x = v for x
        let u = solve_ata_system_enhanced(a, &v, 10)?; // 10 inner iterations

        // Normalize u
        let u_norm = (u.iter().map(|&x| x * x).sum::<T>()).sqrt();
        if u_norm == T::zero() {
            break;
        }

        for i in 0..n {
            v[i] = u[i] / u_norm;
        }

        // Estimate eigenvalue: λ = v^T * (A^T * A) * v using Rayleigh quotient
        lambda = estimate_rayleigh_quotient_enhanced(a, &v)?;

        // Check convergence
        if iter > 0 {
            let rel_change = if lambda != T::zero() {
                ((lambda - lambda_old) / lambda).abs()
            } else {
                T::infinity()
            };

            if rel_change < tol {
                break;
            }
        }

        lambda_old = lambda;
    }

    // The smallest singular value is the square root of the smallest eigenvalue of A^T * A
    Ok(lambda.sqrt())
}

/// Enhanced sparse matrix-vector product
#[allow(dead_code)]
fn sparse_matvec<T, S>(a: &S, x: &ArrayView1<T>) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Mul<Output = T> + 'static,
    S: SparseArray<T>,
{
    let (m, n) = a.shape();
    if x.len() != n {
        return Err(crate::error::SparseError::DimensionMismatch {
            expected: n,
            found: x.len(),
        });
    }

    let mut result = Array1::zeros(m);
    let (row_indices, col_indices, values) = a.find();

    for (k, (&i, &j)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        result[i] = result[i] + values[k] * x[j];
    }

    Ok(result)
}

/// Enhanced sparse matrix-vector product with transpose
#[allow(dead_code)]
fn sparse_matvec_transpose<T, S>(a: &S, x: &ArrayView1<T>) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Mul<Output = T> + 'static,
    S: SparseArray<T>,
{
    let (m, n) = a.shape();
    if x.len() != m {
        return Err(crate::error::SparseError::DimensionMismatch {
            expected: m,
            found: x.len(),
        });
    }

    let mut result = Array1::zeros(n);
    let (row_indices, col_indices, values) = a.find();

    for (k, (&i, &j)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        result[j] = result[j] + values[k] * x[i];
    }

    Ok(result)
}

/// Approximately solve (A^T * A) * x = b using enhanced iterative method
#[allow(dead_code)]
fn solve_ata_system_enhanced<T, S>(
    a: &S,
    b: &Array1<T>,
    num_iterations: usize,
) -> SparseResult<Array1<T>>
where
    T: Float
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + scirs2_core::simd_ops::SimdUnifiedOps
        + Send
        + Sync,
    S: SparseArray<T>,
{
    let (_, n) = a.shape();
    let mut x = b.clone(); // Initial guess

    for _ in 0..num_iterations {
        // Compute r = A^T * A * x
        let ax = sparse_matvec(a, &x.view())?;
        let ata_x = sparse_matvec_transpose(a, &ax.view())?;

        // Simple iteration: x = x - α * (A^T * A * x - b)
        let alpha = T::from(0.1).unwrap(); // Conservative step size
        for i in 0..n {
            x[i] = x[i] - alpha * (ata_x[i] - b[i]);
        }
    }

    Ok(x)
}

/// Estimate the Rayleigh quotient v^T * (A^T * A) * v with enhanced accuracy
#[allow(dead_code)]
fn estimate_rayleigh_quotient_enhanced<T, S>(a: &S, v: &Array1<T>) -> SparseResult<T>
where
    T: Float + Debug + Copy + Add<Output = T> + Mul<Output = T> + 'static + std::iter::Sum,
    S: SparseArray<T>,
{
    // Compute A * v
    let av = sparse_matvec(a, &v.view())?;

    // Compute A^T * (A * v)
    let ata_v = sparse_matvec_transpose(a, &av.view())?;

    // Return v^T * (A^T * A * v)
    Ok(v.iter().zip(ata_v.iter()).map(|(&vi, &ai)| vi * ai).sum())
}

/// Compute exact 2-norm for small sparse arrays
#[allow(dead_code)]
fn exact_twonorm_enhanced<T, S>(a: &S) -> SparseResult<T>
where
    T: Float
        + NumAssign
        + Sum
        + Debug
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + 'static
        + scirs2_core::simd_ops::SimdUnifiedOps
        + Send
        + Sync,
    S: SparseArray<T>,
{
    let (m, n) = a.shape();
    let min_dim = m.min(n);

    if min_dim == 0 {
        return Ok(T::zero());
    }

    if min_dim == 1 {
        // For 1D case, compute the norm of all entries
        let (_, _, values) = a.find();
        let max_val =
            values
                .iter()
                .map(|&v| v.abs())
                .fold(T::zero(), |acc, v| if v > acc { v } else { acc });
        return Ok(max_val);
    }

    // For small matrices, use high-precision power iteration
    twonormest_enhanced(a, Some(T::from(1e-12).unwrap()), Some(1000), None)
}

/// Compute exact 1-norm for small sparse arrays
#[allow(dead_code)]
fn exact_onenorm_enhanced<T, S>(a: &S) -> SparseResult<T>
where
    T: Float + Debug + Copy + Add<Output = T> + 'static,
    S: SparseArray<T>,
{
    let (_, n) = a.shape();
    let mut max_col_sum = T::zero();

    for j in 0..n {
        let mut col_sum = T::zero();
        let (row_indices, col_indices, values) = a.find();

        for (k, (&_i, &col)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
            if col == j {
                col_sum = col_sum + values[k].abs();
            }
        }

        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }

    Ok(max_col_sum)
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
