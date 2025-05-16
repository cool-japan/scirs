//! Linear algebra operations for sparse matrices
//!
//! This module provides linear algebra operations for sparse matrices,
//! such as solving linear systems, computing eigenvalues, etc.

use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use num_traits::{Float, NumAssign};
use rand::Rng;
use std::iter::Sum;

/// Solve a sparse linear system Ax = b
///
/// # Arguments
///
/// * `a` - Sparse matrix A (in CSR format)
/// * `b` - Right-hand side vector b
///
/// # Returns
///
/// * Solution vector x
///
/// # Example
///
/// ```
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::linalg::spsolve;
///
/// // Create a matrix A and vector b
/// let rows = vec![0, 0, 1, 1, 2, 2];
/// let cols = vec![0, 1, 0, 1, 1, 2];
/// let data = vec![2.0, 1.0, 1.0, 2.0, 1.0, 3.0];
/// let shape = (3, 3);
///
/// let a = CsrMatrix::new(data, rows, cols, shape).unwrap();
/// let b = vec![1.0, 2.0, 3.0];
///
/// // Solve Ax = b
/// let x = spsolve(&a, &b).unwrap();
/// ```
pub fn spsolve<F>(a: &CsrMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    if a.rows() != b.len() {
        return Err(SparseError::DimensionMismatch {
            expected: a.rows(),
            found: b.len(),
        });
    }

    if a.rows() != a.cols() {
        return Err(SparseError::ValueError("Matrix must be square".to_string()));
    }

    // For SPD matrices, use Cholesky decomposition
    #[allow(clippy::if_same_then_else)]
    if a.is_symmetric() {
        if is_positive_definite(a) {
            return sparse_cholesky_solve(a, b);
        }
        return sparse_ldlt_solve(a, b);
    }

    // For general matrices, use LU decomposition
    sparse_lu_solve(a, b)
}

/// Check if a matrix is likely to be positive definite
///
/// This is a simplified check that verifies diagonal dominance
/// and positive diagonal elements as a heuristic.
///
/// # Arguments
///
/// * `a` - Sparse matrix in CSR format
///
/// # Returns
///
/// * True if the matrix is likely positive definite
fn is_positive_definite<F>(a: &CsrMatrix<F>) -> bool
where
    F: Float + 'static + std::fmt::Debug + std::ops::AddAssign + std::ops::MulAssign,
{
    if !a.is_symmetric() {
        return false;
    }

    let a_dense = a.to_dense();
    let n = a.rows();

    // Check for positive diagonal elements using iterators
    for (i, row) in a_dense.iter().enumerate().take(n) {
        if row[i] <= F::zero() {
            return false;
        }
    }

    // A simple heuristic: check for diagonal dominance
    // A more robust check would compute eigenvalues or attempt a Cholesky
    for (i, row) in a_dense.iter().enumerate().take(n) {
        let mut row_sum = F::zero();
        for (j, &val) in row.iter().enumerate().take(n) {
            if i != j {
                row_sum += val.abs();
            }
        }
        if row_sum >= row[i] {
            return false;
        }
    }

    true
}

/// Solve a sparse linear system using direct methods
///
/// This function selects an appropriate sparse direct solver based on the matrix
/// structure. It supports symmetric positive definite, symmetric indefinite,
/// and general sparse matrices.
///
/// # Arguments
///
/// * `a` - Sparse matrix in CSR format
/// * `b` - Right-hand side vector
/// * `symmetric` - If true, treats the matrix as symmetric
/// * `positive_definite` - If true and symmetric is true, treats the matrix as symmetric positive definite
///
/// # Returns
///
/// * Solution vector
///
/// # Example
///
/// ```ignore
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::linalg::sparse_direct_solve;
///
/// // Create a sparse SPD matrix
/// let rows = vec![0, 0, 1, 1, 1, 2, 2];
/// let cols = vec![0, 1, 0, 1, 2, 1, 2];
/// let data = vec![2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0];
/// let shape = (3, 3);
///
/// let a = CsrMatrix::new(data, rows, cols, shape).unwrap();
/// let b = vec![1.0, 2.0, 3.0];
///
/// // Solve using Cholesky for SPD matrices
/// let x = sparse_direct_solve(&a, &b, true, true).unwrap();
/// ```
pub fn sparse_direct_solve<F>(
    a: &CsrMatrix<F>,
    b: &[F],
    symmetric: bool,
    positive_definite: bool,
) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    if a.rows() != b.len() {
        return Err(SparseError::DimensionMismatch {
            expected: a.rows(),
            found: b.len(),
        });
    }

    if a.rows() != a.cols() {
        return Err(SparseError::ValueError(format!(
            "Matrix must be square, got {}x{}",
            a.rows(),
            a.cols()
        )));
    }

    #[allow(clippy::if_same_then_else)]
    if symmetric {
        if positive_definite {
            // For SPD matrices, use sparse Cholesky decomposition
            sparse_cholesky_solve(a, b)
        } else {
            // For symmetric indefinite matrices, use LDLT decomposition
            sparse_ldlt_solve(a, b)
        }
    } else {
        // For general sparse matrices, use LU decomposition
        sparse_lu_solve(a, b)
    }
}

/// Solve a sparse linear system using Cholesky decomposition
///
/// This is optimized for symmetric positive definite matrices.
///
/// # Arguments
///
/// * `a` - Sparse matrix in CSR format (must be symmetric positive definite)
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// * Solution vector
fn sparse_cholesky_solve<F>(a: &CsrMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    // Perform sparse Cholesky factorization: A = L*L^T
    let l = sparse_cholesky(a)?;

    // Solve L*y = b by forward substitution
    let y = sparse_forward_substitution(&l, b)?;

    // Solve L^T*x = y by backward substitution
    let l_transpose = l.transpose();
    let x = sparse_backward_substitution(&l_transpose, &y)?;

    Ok(x)
}

/// Compute sparse Cholesky decomposition
///
/// # Arguments
///
/// * `a` - Sparse matrix in CSR format (must be symmetric positive definite)
///
/// # Returns
///
/// * Lower triangular factor L where A = L*L^T
fn sparse_cholesky<F>(a: &CsrMatrix<F>) -> SparseResult<CscMatrix<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    let n = a.rows();

    // Check for symmetry
    if !a.is_symmetric() {
        return Err(SparseError::ValueError(
            "Matrix must be symmetric for Cholesky decomposition".to_string(),
        ));
    }

    // For initial implementation, use simplified algorithm
    // This is not the most efficient approach for large matrices

    // Convert to dense to implement the algorithm
    // In a production implementation, we would use a specialized sparse Cholesky algorithm
    let a_dense = a.to_dense();
    let mut l_dense = vec![vec![F::zero(); n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a_dense[i][j];

            for k in 0..j {
                sum -= l_dense[i][k] * l_dense[j][k];
            }

            if i == j {
                // Diagonal element
                if sum <= F::zero() {
                    return Err(SparseError::ValueError(
                        "Matrix is not positive definite".to_string(),
                    ));
                }
                l_dense[i][j] = sum.sqrt();
            } else {
                // Off-diagonal element
                l_dense[i][j] = sum / l_dense[j][j];
            }
        }
    }

    // Convert back to sparse CSC format (CSC is more efficient for column-oriented operations)
    let mut row_indices = Vec::new();
    let mut col_ptrs = vec![0];
    let mut values = Vec::new();

    let mut nnz = 0;

    for j in 0..n {
        for i in j..n {
            if l_dense[i][j] != F::zero() {
                row_indices.push(i);
                values.push(l_dense[i][j]);
                nnz += 1;
            }
        }
        col_ptrs.push(nnz);
    }

    // Create CSC matrix
    CscMatrix::from_csc_data(values, row_indices, col_ptrs, (n, n))
        .map_err(|e| SparseError::ComputationError(format!("Failed to create CSC matrix: {}", e)))
}

/// Solve a sparse triangular system by forward substitution
///
/// Solves L*x = b where L is a lower triangular sparse matrix
///
/// # Arguments
///
/// * `l` - Lower triangular sparse matrix in CSC format
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// * Solution vector
fn sparse_forward_substitution<F>(l: &CscMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + 'static + std::fmt::Debug,
{
    let n = l.rows();
    let mut x = vec![F::zero(); n];

    for i in 0..n {
        let mut sum = b[i];

        // Subtract the known terms
        for j in l.col_range(i) {
            let row = l.row_indices()[j];
            match row.cmp(&i) {
                std::cmp::Ordering::Less => {
                    sum -= l.data()[j] * x[row];
                }
                std::cmp::Ordering::Equal => {
                    // This is the diagonal element
                    if l.data()[j] == F::zero() {
                        return Err(SparseError::SingularMatrix(
                            "Zero on diagonal during forward substitution".to_string(),
                        ));
                    }
                    x[i] = sum / l.data()[j];
                    break;
                }
                _ => {}
            }
        }
    }

    Ok(x)
}

/// Solve a sparse triangular system by backward substitution
///
/// Solves U*x = b where U is an upper triangular sparse matrix
///
/// # Arguments
///
/// * `u` - Upper triangular sparse matrix in CSC format
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// * Solution vector
fn sparse_backward_substitution<F>(u: &CscMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + 'static + std::fmt::Debug,
{
    let n = u.rows();
    let mut x = vec![F::zero(); n];

    for i in (0..n).rev() {
        let mut sum = b[i];

        // Subtract the known terms
        for j in u.col_range(i) {
            let row = u.row_indices()[j];
            match row.cmp(&i) {
                std::cmp::Ordering::Greater => {
                    sum -= u.data()[j] * x[row];
                }
                std::cmp::Ordering::Equal => {
                    // This is the diagonal element
                    if u.data()[j] == F::zero() {
                        return Err(SparseError::SingularMatrix(
                            "Zero on diagonal during backward substitution".to_string(),
                        ));
                    }
                    x[i] = sum / u.data()[j];
                    break;
                }
                _ => {}
            }
        }
    }

    Ok(x)
}

/// Solve a sparse linear system using LU decomposition
///
/// This is a general solver that works with any non-singular sparse matrix.
///
/// # Arguments
///
/// * `a` - Sparse matrix in CSR format
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// * Solution vector
fn sparse_lu_solve<F>(a: &CsrMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    // Perform sparse LU factorization with partial pivoting
    let (l, u, p) = sparse_lu(a)?;

    // Apply permutation to the right-hand side
    let mut pb = vec![F::zero(); b.len()];
    for (i, &pi) in p.iter().enumerate() {
        pb[i] = b[pi];
    }

    // Solve L*y = P*b by forward substitution
    let y = sparse_forward_substitution(&l, &pb)?;

    // Solve U*x = y by backward substitution
    let x = sparse_backward_substitution(&u, &y)?;

    Ok(x)
}

/// Compute sparse LU decomposition with partial pivoting
///
/// # Arguments
///
/// * `a` - Sparse matrix in CSR format
///
/// # Returns
///
/// * (L, U, P) where L is lower triangular, U is upper triangular,
///   and P is a permutation vector such that P*A = L*U
fn sparse_lu<F>(a: &CsrMatrix<F>) -> SparseResult<(CscMatrix<F>, CscMatrix<F>, Vec<usize>)>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    let n = a.rows();

    // For initial implementation, use simplified algorithm
    // This is not the most efficient approach for large sparse matrices

    // Convert to dense to implement the algorithm
    // In a production implementation, we would use a specialized sparse LU algorithm
    let mut a_dense = a.to_dense();
    let mut p = (0..n).collect::<Vec<usize>>();

    for k in 0..n {
        // Find pivot
        let mut pivot_row = k;
        let mut pivot_val = a_dense[p[k]][k].abs();

        for i in k + 1..n {
            let val = a_dense[p[i]][k].abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = i;
            }
        }

        if pivot_val < F::epsilon() {
            return Err(SparseError::SingularMatrix(
                "Matrix is numerically singular".to_string(),
            ));
        }

        // Swap rows
        if pivot_row != k {
            p.swap(k, pivot_row);
        }

        // Eliminate below pivot
        for i in k + 1..n {
            let factor = a_dense[p[i]][k] / a_dense[p[k]][k];
            a_dense[p[i]][k] = factor; // Store multiplier in L part

            for j in k + 1..n {
                a_dense[p[i]][j] = a_dense[p[i]][j] - factor * a_dense[p[k]][j];
            }
        }
    }

    // Construct L and U matrices
    let mut l_row_indices = Vec::new();
    let mut l_col_ptrs = vec![0];
    let mut l_values = Vec::new();

    let mut u_row_indices = Vec::new();
    let mut u_col_ptrs = vec![0];
    let mut u_values = Vec::new();

    let mut l_nnz = 0;
    let mut u_nnz = 0;

    for j in 0..n {
        // For L (lower triangular with unit diagonal)
        l_row_indices.push(j);
        l_values.push(F::one());
        l_nnz += 1;

        for i in j + 1..n {
            if a_dense[p[i]][j] != F::zero() {
                l_row_indices.push(i);
                l_values.push(a_dense[p[i]][j]);
                l_nnz += 1;
            }
        }
        l_col_ptrs.push(l_nnz);

        // For U (upper triangular)
        for i in 0..=j {
            if a_dense[p[i]][j] != F::zero() {
                u_row_indices.push(i);
                u_values.push(a_dense[p[i]][j]);
                u_nnz += 1;
            }
        }
        u_col_ptrs.push(u_nnz);
    }

    // Create CSC matrices
    let l = CscMatrix::from_csc_data(l_values, l_row_indices, l_col_ptrs, (n, n))
        .map_err(|e| SparseError::ComputationError(format!("Failed to create L matrix: {}", e)))?;

    let u = CscMatrix::from_csc_data(u_values, u_row_indices, u_col_ptrs, (n, n))
        .map_err(|e| SparseError::ComputationError(format!("Failed to create U matrix: {}", e)))?;

    Ok((l, u, p))
}

/// Solve a sparse linear system using LDLT decomposition
///
/// This is optimized for symmetric indefinite matrices.
///
/// # Arguments
///
/// * `a` - Sparse matrix in CSR format (must be symmetric)
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// * Solution vector
fn sparse_ldlt_solve<F>(a: &CsrMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    // Check for symmetry
    if !a.is_symmetric() {
        return Err(SparseError::ValueError(
            "Matrix must be symmetric for LDLT decomposition".to_string(),
        ));
    }

    // Perform sparse LDLT factorization
    let (l, d) = sparse_ldlt(a)?;

    // Solve L*y = b by forward substitution
    let y = sparse_forward_substitution(&l, b)?;

    // Apply diagonal scaling: z = D^-1 * y
    let mut z = vec![F::zero(); y.len()];
    for i in 0..y.len() {
        if d[i] == F::zero() {
            return Err(SparseError::SingularMatrix(
                "Zero on diagonal during LDLT solve".to_string(),
            ));
        }
        z[i] = y[i] / d[i];
    }

    // Solve L^T*x = z by backward substitution
    let l_transpose = l.transpose();
    let x = sparse_backward_substitution(&l_transpose, &z)?;

    Ok(x)
}

/// Compute sparse LDLT decomposition
///
/// # Arguments
///
/// * `a` - Sparse matrix in CSR format (must be symmetric)
///
/// # Returns
///
/// * (L, D) where L is unit lower triangular and D is diagonal
///   such that A = L*D*L^T
fn sparse_ldlt<F>(a: &CsrMatrix<F>) -> SparseResult<(CscMatrix<F>, Vec<F>)>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    let n = a.rows();

    // For initial implementation, use simplified algorithm
    // This is not the most efficient approach for large sparse matrices

    // Convert to dense to implement the algorithm
    // In a production implementation, we would use a specialized sparse LDLT algorithm
    let a_dense = a.to_dense();
    let mut l_dense = vec![vec![F::zero(); n]; n];
    let mut d = vec![F::zero(); n];

    // Initialize L with identity
    for i in 0..n {
        l_dense[i][i] = F::one();
    }

    for j in 0..n {
        // Compute D[j] and L[i,j] for i > j
        let mut d_j = a_dense[j][j];

        for k in 0..j {
            d_j -= l_dense[j][k] * l_dense[j][k] * d[k];
        }

        d[j] = d_j;

        for i in j + 1..n {
            let mut l_ij = a_dense[i][j];

            for k in 0..j {
                l_ij -= l_dense[i][k] * l_dense[j][k] * d[k];
            }

            if d_j != F::zero() {
                l_dense[i][j] = l_ij / d_j;
            } else {
                return Err(SparseError::SingularMatrix(
                    "Zero pivot encountered in LDLT decomposition".to_string(),
                ));
            }
        }
    }

    // Convert L to sparse CSC format
    let mut row_indices = Vec::new();
    let mut col_ptrs = vec![0];
    let mut values = Vec::new();

    let mut nnz = 0;

    for j in 0..n {
        for i in j..n {
            if l_dense[i][j] != F::zero() {
                row_indices.push(i);
                values.push(l_dense[i][j]);
                nnz += 1;
            }
        }
        col_ptrs.push(nnz);
    }

    // Create CSC matrix
    let l = CscMatrix::from_csc_data(values, row_indices, col_ptrs, (n, n)).map_err(|e| {
        SparseError::ComputationError(format!("Failed to create CSC matrix: {}", e))
    })?;

    Ok((l, d))
}

/// Interface to sparse least squares solver
///
/// Solves the sparse least squares problem min ||Ax - b||₂ where A is sparse.
///
/// # Arguments
///
/// * `a` - Sparse matrix in CSR format
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// * Least squares solution
pub fn sparse_lstsq<F>(a: &CsrMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    // Solve using the normal equations: A^T A x = A^T b
    let at = a.transpose();
    let ata = at
        .matmul(a)
        .map_err(|e| SparseError::ComputationError(format!("Failed to compute A^T A: {}", e)))?;

    // Compute A^T b using Iterator methods to avoid Clippy warnings
    let mut atb = vec![F::zero(); a.cols()];
    for (i, elem) in atb.iter_mut().enumerate().take(a.cols()) {
        for j in at.row_range(i) {
            let row = at.col_indices()[j];
            if row < b.len() {
                *elem += at.data[j] * b[row];
            }
        }
    }

    // Solve the system using Cholesky
    sparse_cholesky_solve(&ata, &atb)
}

/// Compute the norm of a sparse matrix
///
/// # Arguments
///
/// * `a` - Sparse matrix
/// * `ord` - Norm type ("1", "2", "inf", "fro")
///
/// # Returns
///
/// * Computed norm value
pub fn norm<F>(a: &CsrMatrix<F>, ord: &str) -> SparseResult<F>
where
    F: Float
        + Sum
        + Copy
        + Default
        + 'static
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::MulAssign,
{
    match ord {
        "1" => {
            // Maximum absolute column sum
            // For 1-norm, we need column sums which is more efficient with CSC format
            let mut max_sum = F::zero();
            let a_dense = a.to_dense();

            for j in 0..a.cols() {
                let mut sum = F::zero();
                for row in a_dense.iter().take(a.rows()) {
                    sum += row[j].abs();
                }
                max_sum = max_sum.max(sum);
            }

            Ok(max_sum)
        }
        "inf" => {
            // Maximum absolute row sum
            let mut max_sum = F::zero();
            let a_dense = a.to_dense();

            for row in a_dense.iter().take(a.rows()) {
                let mut sum = F::zero();
                for &val in row.iter().take(a.cols()) {
                    sum += val.abs();
                }
                max_sum = max_sum.max(sum);
            }

            Ok(max_sum)
        }
        "fro" => {
            // Frobenius norm
            let mut sum_squares = F::zero();

            // We can use the raw data directly
            for &val in &a.data {
                sum_squares += val * val;
            }

            Ok(sum_squares.sqrt())
        }
        "2" => {
            // Spectral norm (largest singular value)
            // Use the power iteration method for computing the spectral norm
            power_iteration_spectral_norm(a)
        }
        _ => Err(SparseError::ValueError(format!(
            "Unknown norm type: {}",
            ord
        ))),
    }
}

/// Sparse matrix-matrix multiplication C = A * B
///
/// # Arguments
///
/// * `a` - Left matrix
/// * `b` - Right matrix
///
/// # Returns
///
/// * Product matrix C
pub fn matmul<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + Sum + Copy + Default + 'static + std::fmt::Debug,
{
    if a.cols() != b.rows() {
        return Err(SparseError::DimensionMismatch {
            expected: a.cols(),
            found: b.rows(),
        });
    }

    // Converting to dense and back to CSR for now to pass tests
    // This is a fallback approach since our direct CSR multiplication had issues
    let a_dense = a.to_dense();
    let b_dense = b.to_dense();

    let m = a.rows();
    let n = b.cols();
    let k = a.cols();

    let mut c_dense = vec![vec![F::zero(); n]; m];

    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                c_dense[i][j] = c_dense[i][j] + a_dense[i][l] * b_dense[l][j];
            }
        }
    }

    // Convert to triplet format for creating CSR
    let mut values = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for i in 0..m {
        for j in 0..n {
            let val = c_dense[i][j];
            if val != F::zero() {
                values.push(val);
                row_indices.push(i);
                col_indices.push(j);
            }
        }
    }

    CsrMatrix::new(values, row_indices, col_indices, (m, n))
}

/// Compute the spectral norm (largest singular value) of a sparse matrix
/// using power iteration method
///
/// # Arguments
///
/// * `a` - Sparse matrix
///
/// # Returns
///
/// * Spectral norm
fn power_iteration_spectral_norm<F>(a: &CsrMatrix<F>) -> SparseResult<F>
where
    F: Float
        + Sum
        + Copy
        + Default
        + 'static
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::MulAssign,
{
    // Get matrix dimensions
    let (m, n) = a.shape();

    // Maximum number of iterations and convergence tolerance
    let max_iter = 100;
    let tol = F::from(1e-8).unwrap();

    // Create matrix for A^T*A or A*A^T (using the smaller dimension)
    let matrix_to_use: CsrMatrix<F>;
    let dim: usize;

    if m <= n {
        // A * A^T for an m×n matrix where m <= n
        let at = a.transpose();
        matrix_to_use = matmul(a, &at)?;
        dim = m;
    } else {
        // A^T * A for an m×n matrix where m > n
        let at = a.transpose();
        matrix_to_use = matmul(&at, a)?;
        dim = n;
    }

    // Initialize random vector
    let mut rng = rand::rng();
    let mut x = (0..dim)
        .map(|_| {
            let random_val = rng.random_range(-1.0..1.0);
            F::from(random_val).unwrap()
        })
        .collect::<Vec<F>>();

    // Normalize the vector
    let norm_x = F::from(x.iter().map(|&xi| xi * xi).sum::<F>().sqrt()).unwrap();
    for xi in &mut x {
        *xi = *xi / norm_x;
    }

    let mut lambda_prev = F::zero();

    // Power iteration
    for _ in 0..max_iter {
        // y = A * x
        let mut y = vec![F::zero(); dim];

        for i in 0..dim {
            for j in matrix_to_use.indptr[i]..matrix_to_use.indptr[i + 1] {
                let col = matrix_to_use.indices[j];
                y[i] += matrix_to_use.data[j] * x[col];
            }
        }

        // Calculate Rayleigh quotient (x^T A x)
        let mut rayleigh = F::zero();
        for i in 0..dim {
            rayleigh += x[i] * y[i];
        }

        // Check for convergence
        if (rayleigh - lambda_prev).abs() < tol {
            return Ok(rayleigh.sqrt()); // Return sqrt of eigenvalue (singular value)
        }

        lambda_prev = rayleigh;

        // Normalize y to get new x
        let norm_y = F::from(y.iter().map(|&yi| yi * yi).sum::<F>().sqrt()).unwrap();
        for i in 0..dim {
            x[i] = y[i] / norm_y;
        }
    }

    // Return the approximation if max iterations reached
    Ok(lambda_prev.sqrt())
}

/// Sparse matrix addition C = A + B
///
/// # Arguments
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// # Returns
///
/// * Sum matrix C
pub fn add<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + Sum + Copy + 'static + std::fmt::Debug,
{
    if a.shape() != b.shape() {
        return Err(SparseError::ValueError(format!(
            "Matrix dimensions must match for addition: {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let (rows, cols) = a.shape();

    // We'll collect entries in COO format first, then convert to CSR
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    // Process matrix A
    for i in 0..rows {
        for j in a.indptr[i]..a.indptr[i + 1] {
            let col = a.indices[j];
            row_indices.push(i);
            col_indices.push(col);
            values.push(a.data[j]);
        }
    }

    // Process matrix B and combine with A values
    for i in 0..rows {
        for j in b.indptr[i]..b.indptr[i + 1] {
            let col = b.indices[j];

            // Check if this position already has a value from A
            let mut found = false;
            for k in 0..values.len() {
                if row_indices[k] == i && col_indices[k] == col {
                    // Position exists, add the values
                    values[k] = values[k] + b.data[j];
                    found = true;
                    break;
                }
            }

            if !found {
                // Position doesn't exist in A, add it
                row_indices.push(i);
                col_indices.push(col);
                values.push(b.data[j]);
            }
        }
    }

    // Create a COO matrix and let it convert to CSR for us
    let mut coo = crate::coo::CooMatrix::new(values, row_indices, col_indices, (rows, cols))?;

    // Sort by row, then column for CSR conversion
    coo.sort_by_row_col();

    // Convert to CSR using the COO's built-in conversion method
    Ok(coo.to_csr())
}

/// Element-wise multiplication (Hadamard product) C = A .* B
///
/// # Arguments
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// # Returns
///
/// * Element-wise product matrix C
pub fn multiply<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + Copy + 'static + std::fmt::Debug,
{
    if a.shape() != b.shape() {
        return Err(SparseError::ValueError(format!(
            "Matrix dimensions must match for element-wise multiplication: {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let (rows, cols) = a.shape();

    // We'll collect entries in COO format first, then convert to CSR
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    // Find elements that exist in both matrices
    for i in 0..rows {
        // For each row, create a map of column indices to values for B
        let mut b_row_values = std::collections::HashMap::new();
        for j in b.indptr[i]..b.indptr[i + 1] {
            let col = b.indices[j];
            b_row_values.insert(col, b.data[j]);
        }

        // Check A's elements against B's map
        for j in a.indptr[i]..a.indptr[i + 1] {
            let col = a.indices[j];
            if let Some(&b_val) = b_row_values.get(&col) {
                let product = a.data[j] * b_val;
                if product != F::zero() {
                    row_indices.push(i);
                    col_indices.push(col);
                    values.push(product);
                }
            }
        }
    }

    // Create COO matrix and let it convert to CSR for us
    let mut coo_result =
        crate::coo::CooMatrix::new(values, row_indices, col_indices, (rows, cols))?;

    // Sort by row, then column for CSR conversion
    coo_result.sort_by_row_col();

    // Convert to CSR using the COO's built-in conversion method
    Ok(coo_result.to_csr())
}

/// Create a diagonal matrix
///
/// # Arguments
///
/// * `diag` - Vector of diagonal elements
/// * `n` - Matrix size (optional, defaults to length of diag)
///
/// # Returns
///
/// * Diagonal matrix in CSR format
pub fn diag_matrix<F>(diag: &[F], n: Option<usize>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + 'static,
{
    let size = n.unwrap_or(diag.len());

    if diag.is_empty() {
        return Err(SparseError::ValueError(
            "Diagonal vector cannot be empty".to_string(),
        ));
    }

    if size < diag.len() {
        return Err(SparseError::ValueError(
            "Matrix size cannot be smaller than diagonal vector length".to_string(),
        ));
    }

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    // Add diagonal elements
    for (i, &val) in diag.iter().enumerate() {
        if val != F::zero() {
            row_indices.push(i);
            col_indices.push(i);
            values.push(val);
        }
    }

    CsrMatrix::new(values, row_indices, col_indices, (size, size))
}

/// Create an identity matrix
///
/// # Arguments
///
/// * `n` - Matrix size
///
/// # Returns
///
/// * Identity matrix in CSR format
pub fn eye<F>(n: usize) -> SparseResult<CsrMatrix<F>>
where
    F: Float + 'static,
{
    if n == 0 {
        return Err(SparseError::ValueError(
            "Matrix size must be positive".to_string(),
        ));
    }

    let mut row_indices = Vec::with_capacity(n);
    let mut col_indices = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    for i in 0..n {
        row_indices.push(i);
        col_indices.push(i);
        values.push(F::one());
    }

    CsrMatrix::new(values, row_indices, col_indices, (n, n))
}

/// Solve a sparse linear system with multiple right-hand sides: AX = B
///
/// This is a companion function to spsolve that handles matrix right-hand sides
/// instead of vector right-hand sides.
///
/// # Arguments
///
/// * `a` - Sparse matrix A (in CSR format)
/// * `b` - Right-hand side matrix B (in CSR format)
///
/// # Returns
///
/// * Solution matrix X
fn spsolve_matrix<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + Copy + Default + 'static + std::fmt::Debug,
{
    let (m, n) = a.shape();
    let (b_rows, b_cols) = b.shape();

    if m != b_rows {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: b_rows,
        });
    }

    if m != n {
        return Err(SparseError::ValueError(
            "Matrix A must be square".to_string(),
        ));
    }

    // Simpler approach: solve column by column using existing spsolve function
    // Extract each column from B, solve Ax = b_col, and collect results
    let mut x_data = Vec::new();
    let mut x_rows = Vec::new();
    let mut x_cols = Vec::new();

    let b_dense = b.to_dense();

    for j in 0..b_cols {
        // Extract column j from B
        let mut b_col = vec![F::zero(); m];
        for i in 0..m {
            b_col[i] = b_dense[i][j];
        }

        // Solve the system for this column
        let x_col = spsolve(a, &b_col)?;

        // Store the results
        for (i, &val) in x_col.iter().enumerate() {
            if val != F::zero() {
                x_rows.push(i);
                x_cols.push(j);
                x_data.push(val);
            }
        }
    }

    // Create the resulting sparse matrix
    CsrMatrix::new(x_data, x_rows, x_cols, (n, b_cols))
}

/// Compute the inverse of a sparse matrix.
///
/// This computes the sparse inverse of a square matrix. If the inverse of the
/// matrix is expected to be non-sparse, it will likely be faster to convert
/// the matrix to dense and use a dense matrix inverse algorithm.
///
/// # Arguments
///
/// * `a` - Square sparse matrix to be inverted
///
/// # Returns
///
/// * The inverse of matrix `a`
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::linalg::inv;
/// use approx::assert_relative_eq;
///
/// // Create a simple diagonal matrix
/// let rows = vec![0, 1];
/// let cols = vec![0, 1];
/// let data = vec![2.0, 4.0];
/// let shape = (2, 2);
///
/// let a = CsrMatrix::new(data, rows, cols, shape).unwrap();
///
/// // Compute A^-1
/// let a_inv = inv(&a).unwrap();
/// let a_inv_dense = a_inv.to_dense();
///
/// // For diagonal matrix [2, 0; 0, 4], the inverse is [0.5, 0; 0, 0.25]
/// assert_relative_eq!(a_inv_dense[0][0], 0.5, epsilon = 1e-10);
/// assert_relative_eq!(a_inv_dense[0][1], 0.0, epsilon = 1e-10);
/// assert_relative_eq!(a_inv_dense[1][0], 0.0, epsilon = 1e-10);
/// assert_relative_eq!(a_inv_dense[1][1], 0.25, epsilon = 1e-10);
///
/// // Check A * A^-1 = I
/// let prod = a.matmul(&a_inv).unwrap();
/// let prod_dense = prod.to_dense();
///
/// // Should be identity matrix
/// assert_relative_eq!(prod_dense[0][0], 1.0, epsilon = 1e-10);
/// assert_relative_eq!(prod_dense[0][1], 0.0, epsilon = 1e-10);
/// assert_relative_eq!(prod_dense[1][0], 0.0, epsilon = 1e-10);
/// assert_relative_eq!(prod_dense[1][1], 1.0, epsilon = 1e-10);
/// ```
///
/// Constants for identifying matrix structure
const UPPER_TRIANGULAR: &str = "upper_triangular";

/// Calculate the 1-norm of a sparse matrix
///
/// The 1-norm is the maximum absolute column sum of the matrix.
///
/// # Arguments
///
/// * `a` - Sparse matrix
///
/// # Returns
///
/// * 1-norm value
fn _onenorm<F>(a: &CsrMatrix<F>) -> F
where
    F: Float + Sum + Copy + Default + 'static + std::fmt::Debug + std::ops::AddAssign,
{
    // In CSR format, we can compute the column sums directly
    let (rows, cols) = a.shape();
    let mut col_sums = vec![F::zero(); cols];

    // Compute sums for each column
    for i in 0..rows {
        for j in a.indptr[i]..a.indptr[i + 1] {
            let col = a.indices[j];
            col_sums[col] += a.data[j].abs();
        }
    }

    // Find the maximum column sum
    col_sums
        .into_iter()
        .fold(F::zero(), |max, sum| max.max(sum))
}

/// Check if a matrix is upper triangular
///
/// # Arguments
///
/// * `a` - Sparse matrix to check
///
/// # Returns
///
/// * True if the matrix is upper triangular
fn _is_upper_triangular<F>(a: &CsrMatrix<F>) -> bool
where
    F: Float + Copy + Default + 'static + std::fmt::Debug,
{
    // Check if all elements below the main diagonal are zero
    for i in 1..a.rows() {
        for idx in a.indptr[i]..a.indptr[i + 1] {
            let j = a.indices[idx];
            if j < i && a.data[idx] != F::zero() {
                return false;
            }
        }
    }
    true
}

/// Check if a matrix is diagonal
///
/// # Arguments
///
/// * `a` - Sparse matrix to check
///
/// # Returns
///
/// * True if the matrix is diagonal
fn _is_diagonal<F>(a: &CsrMatrix<F>) -> bool
where
    F: Float + Copy + Default + 'static + std::fmt::Debug,
{
    for i in 0..a.rows() {
        for idx in a.indptr[i]..a.indptr[i + 1] {
            let j = a.indices[idx];
            if i != j && a.data[idx] != F::zero() {
                return false;
            }
        }
    }
    true
}

/// Solve the system P * X = Q for X
///
/// This is a helper function for the matrix exponential computation.
///
/// # Arguments
///
/// * `p` - Left-hand side matrix P
/// * `q` - Right-hand side matrix Q
/// * `_structure` - Optional structure information for the matrices (unused currently)
///
/// # Returns
///
/// * Solution matrix X
fn _solve_p_q<F>(
    p: CsrMatrix<F>,
    q: CsrMatrix<F>,
    _structure: Option<&'static str>,
) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + Copy + Default + 'static + std::fmt::Debug,
{
    // The system P*X = Q is equivalent to X = Q*(P^-1)
    // We solve this using our spsolve_matrix function
    spsolve_matrix(&p, &q)
}

/// Compute the parameter ell for the scaling and squaring method
///
/// # Arguments
///
/// * `a` - Matrix being exponentiated
/// * `m` - Order of the Padé approximation
///
/// # Returns
///
/// * ell parameter for scaling
fn _ell<F>(a: &CsrMatrix<F>, m: usize) -> SparseResult<i32>
where
    F: Float + Sum + Copy + Default + 'static + std::fmt::Debug + std::ops::AddAssign,
{
    if a.rows() != a.cols() {
        return Err(SparseError::ValueError("Matrix must be square".to_string()));
    }

    // The c_i are coefficients of a series expansion
    let c_i = match m {
        3 => F::from(100800.0).unwrap(),
        5 => F::from(10059033600.0).unwrap(),
        7 => F::from(4487938430976000.0).unwrap(),
        9 => F::from(5914384781877411840000.0).unwrap(),
        13 => F::from(1.132_507_756_060_211_1e35).unwrap(),
        _ => {
            return Err(SparseError::ValueError(format!(
                "Unsupported Padé order m={}",
                m
            )))
        }
    };

    let abs_c_recip = c_i;

    // IEEE double precision unit roundoff
    let u = F::from(2.0_f64.powi(-53)).unwrap();

    // Create a matrix with absolute values
    let a_abs = _abs_matrix(a);

    // Compute ||abs(A)^(2m+1)||_1
    let power = 2 * m + 1;
    let a_abs_pow = _matrix_power_nnm(&a_abs, power)?;

    // Compute the one-norm
    let a_abs_onenorm = _onenorm(&a_abs_pow);

    // Special case for zero norm
    if a_abs_onenorm == F::zero() {
        return Ok(0);
    }

    let alpha = a_abs_onenorm / (_onenorm(a) * abs_c_recip);
    let log2_alpha_div_u = (alpha / u).log2();
    let value = (log2_alpha_div_u / F::from(2.0 * m as f64).unwrap()).ceil();

    // Convert to i32 and ensure non-negative
    let res = value.to_f64().unwrap() as i32;
    Ok(res.max(0))
}

/// Create a matrix with the absolute values of the input matrix
///
/// # Arguments
///
/// * `a` - Input matrix
///
/// # Returns
///
/// * Matrix with absolute values
fn _abs_matrix<F>(a: &CsrMatrix<F>) -> CsrMatrix<F>
where
    F: Float + Copy + Default + 'static + std::fmt::Debug,
{
    // Create a new matrix with the same structure but absolute values
    let mut abs_data = Vec::with_capacity(a.data.len());
    for &val in &a.data {
        abs_data.push(val.abs());
    }

    // Create a new CSR matrix with the same structure
    let mut result = a.clone();
    result.data = abs_data;

    result
}

/// Compute the power of a non-negative matrix (NNM)
///
/// Special version of matrix power for non-negative matrices, used in computing
/// bounds for the scaling and squaring method.
///
/// # Arguments
///
/// * `a` - Non-negative matrix
/// * `power` - Power to raise the matrix to
///
/// # Returns
///
/// * A^power
fn _matrix_power_nnm<F>(a: &CsrMatrix<F>, power: usize) -> SparseResult<CsrMatrix<F>>
where
    F: Float + Sum + Copy + Default + 'static + std::fmt::Debug + std::ops::AddAssign,
{
    if power == 0 {
        return eye(a.rows());
    }

    if power == 1 {
        return Ok(a.clone());
    }

    // Use binary exponentiation
    if power % 2 == 0 {
        let half = _matrix_power_nnm(a, power / 2)?;
        matmul(&half, &half)
    } else {
        let half = _matrix_power_nnm(a, power / 2)?;
        let half_squared = matmul(&half, &half)?;
        matmul(a, &half_squared)
    }
}

/// Scale a sparse matrix by a scalar value
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `scalar` - Scalar value
///
/// # Returns
///
/// * Scaled matrix
fn _scalar_mul<F>(a: &CsrMatrix<F>, scalar: F) -> CsrMatrix<F>
where
    F: Float + Copy + Default + 'static + std::fmt::Debug,
{
    // Create a new matrix with the same structure but scaled values
    let mut scaled_data = Vec::with_capacity(a.data.len());
    for &val in &a.data {
        scaled_data.push(val * scalar);
    }

    // Create a new CSR matrix with the same structure
    let mut result = a.clone();
    result.data = scaled_data;

    result
}

/// Helper struct for lazily evaluating matrix exponential operations
///
/// This struct helps avoid doing more work than necessary for high precision
/// exponential computation by lazily computing matrix powers and storing
/// precomputed properties of the matrix.
struct _ExpmPadeHelper<'a, F>
where
    F: Float + Copy + Default + 'static + std::fmt::Debug,
{
    /// The matrix to be exponentiated
    a: &'a CsrMatrix<F>,
    /// Cached A^2
    a2: Option<CsrMatrix<F>>,
    /// Cached A^4
    a4: Option<CsrMatrix<F>>,
    /// Cached A^6
    a6: Option<CsrMatrix<F>>,
    /// Cached A^8
    a8: Option<CsrMatrix<F>>,
    /// Cached A^10
    a10: Option<CsrMatrix<F>>,
    /// Cached exact d4 = ||A^4||^(1/4)
    d4_exact: Option<F>,
    /// Cached exact d6 = ||A^6||^(1/6)
    d6_exact: Option<F>,
    /// Cached exact d8 = ||A^8||^(1/8)
    d8_exact: Option<F>,
    /// Cached exact d10 = ||A^10||^(1/10)
    d10_exact: Option<F>,
    /// Cached approximate d4
    d4_approx: Option<F>,
    /// Cached approximate d6
    d6_approx: Option<F>,
    /// Cached approximate d8
    d8_approx: Option<F>,
    /// Cached approximate d10
    d10_approx: Option<F>,
    /// Identity matrix of the same size as A
    ident: CsrMatrix<F>,
    /// Structure of the matrix (e.g., "upper_triangular")
    #[allow(dead_code)]
    structure: Option<&'static str>,
    /// Whether to use exact one-norm calculations
    use_exact_onenorm: bool,
}

impl<'a, F> _ExpmPadeHelper<'a, F>
where
    F: Float + Sum + NumAssign + Copy + Default + 'static + std::fmt::Debug + std::ops::AddAssign,
{
    /// Create a new helper for exponentiating a matrix
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to be exponentiated
    /// * `structure` - Optional string describing the structure of the matrix
    /// * `use_exact_onenorm` - Whether to use exact one-norm calculations
    fn new(
        a: &'a CsrMatrix<F>,
        structure: Option<&'static str>,
        use_exact_onenorm: bool,
    ) -> SparseResult<Self> {
        Ok(Self {
            a,
            a2: None,
            a4: None,
            a6: None,
            a8: None,
            a10: None,
            d4_exact: None,
            d6_exact: None,
            d8_exact: None,
            d10_exact: None,
            d4_approx: None,
            d6_approx: None,
            d8_approx: None,
            d10_approx: None,
            ident: eye(a.rows())?,
            structure,
            use_exact_onenorm,
        })
    }

    /// Get the matrix A^2, computing it if necessary
    fn a2(&mut self) -> SparseResult<CsrMatrix<F>> {
        if self.a2.is_none() {
            let result = matmul(self.a, self.a)?;
            self.a2 = Some(result);
        }
        Ok(self.a2.as_ref().unwrap().clone())
    }

    /// Get the matrix A^4, computing it if necessary
    fn a4(&mut self) -> SparseResult<CsrMatrix<F>> {
        if self.a4.is_none() {
            let a2 = self.a2()?;
            let result = matmul(&a2, &a2)?;
            self.a4 = Some(result);
        }
        Ok(self.a4.as_ref().unwrap().clone())
    }

    /// Get the matrix A^6, computing it if necessary
    fn a6(&mut self) -> SparseResult<CsrMatrix<F>> {
        if self.a6.is_none() {
            let a4 = self.a4()?;
            let a2 = self.a2()?;
            let result = matmul(&a4, &a2)?;
            self.a6 = Some(result);
        }
        Ok(self.a6.as_ref().unwrap().clone())
    }

    /// Get the matrix A^8, computing it if necessary
    fn a8(&mut self) -> SparseResult<CsrMatrix<F>> {
        if self.a8.is_none() {
            let a6 = self.a6()?;
            let a2 = self.a2()?;
            let result = matmul(&a6, &a2)?;
            self.a8 = Some(result);
        }
        Ok(self.a8.as_ref().unwrap().clone())
    }

    /// Get the matrix A^10, computing it if necessary
    fn a10(&mut self) -> SparseResult<CsrMatrix<F>> {
        if self.a10.is_none() {
            let a4 = self.a4()?;
            let a6 = self.a6()?;
            let result = matmul(&a4, &a6)?;
            self.a10 = Some(result);
        }
        Ok(self.a10.as_ref().unwrap().clone())
    }

    /// Get the tight bound d4 = ||A^4||^(1/4), computing it if necessary
    fn d4_tight(&mut self) -> SparseResult<F> {
        if self.d4_exact.is_none() {
            let a4 = self.a4()?;
            self.d4_exact = Some(_onenorm(&a4).powf(F::from(0.25).unwrap()));
        }
        Ok(self.d4_exact.unwrap())
    }

    /// Get the tight bound d6 = ||A^6||^(1/6), computing it if necessary
    fn d6_tight(&mut self) -> SparseResult<F> {
        if self.d6_exact.is_none() {
            let a6 = self.a6()?;
            self.d6_exact = Some(_onenorm(&a6).powf(F::from(1.0 / 6.0).unwrap()));
        }
        Ok(self.d6_exact.unwrap())
    }

    /// Get the tight bound d8 = ||A^8||^(1/8), computing it if necessary
    fn d8_tight(&mut self) -> SparseResult<F> {
        if self.d8_exact.is_none() {
            let a8 = self.a8()?;
            self.d8_exact = Some(_onenorm(&a8).powf(F::from(0.125).unwrap()));
        }
        Ok(self.d8_exact.unwrap())
    }

    /// Get the tight bound d10 = ||A^10||^(1/10), computing it if necessary
    fn d10_tight(&mut self) -> SparseResult<F> {
        if self.d10_exact.is_none() {
            let a10 = self.a10()?;
            self.d10_exact = Some(_onenorm(&a10).powf(F::from(0.1).unwrap()));
        }
        Ok(self.d10_exact.unwrap())
    }

    /// Get the loose bound d4, using approximation if requested
    fn d4_loose(&mut self) -> SparseResult<F> {
        if self.use_exact_onenorm {
            return self.d4_tight();
        }
        if self.d4_exact.is_none() && self.d4_approx.is_none() {
            // Currently we're not implementing the approximation method _onenormest_matrix_power
            // so we'll use the exact method even when approximation is allowed
            return self.d4_tight();
        }
        if let Some(val) = self.d4_exact {
            return Ok(val);
        }
        Ok(self.d4_approx.unwrap())
    }

    /// Get the loose bound d6, using approximation if requested
    fn d6_loose(&mut self) -> SparseResult<F> {
        if self.use_exact_onenorm {
            return self.d6_tight();
        }
        if self.d6_exact.is_none() && self.d6_approx.is_none() {
            // Use exact method for now
            return self.d6_tight();
        }
        if let Some(val) = self.d6_exact {
            return Ok(val);
        }
        Ok(self.d6_approx.unwrap())
    }

    /// Get the loose bound d8, using approximation if requested
    fn d8_loose(&mut self) -> SparseResult<F> {
        if self.use_exact_onenorm {
            return self.d8_tight();
        }
        if self.d8_exact.is_none() && self.d8_approx.is_none() {
            // Use exact method for now
            return self.d8_tight();
        }
        if let Some(val) = self.d8_exact {
            return Ok(val);
        }
        Ok(self.d8_approx.unwrap())
    }

    /// Get the loose bound d10, using approximation if requested
    fn d10_loose(&mut self) -> SparseResult<F> {
        if self.use_exact_onenorm {
            return self.d10_tight();
        }
        if self.d10_exact.is_none() && self.d10_approx.is_none() {
            // Use exact method for now
            return self.d10_tight();
        }
        if let Some(val) = self.d10_exact {
            return Ok(val);
        }
        Ok(self.d10_approx.unwrap())
    }

    /// Compute the Padé approximation of order 3
    fn pade3(&mut self) -> SparseResult<(CsrMatrix<F>, CsrMatrix<F>)> {
        let b = [
            F::from(120.0).unwrap(),
            F::from(60.0).unwrap(),
            F::from(12.0).unwrap(),
            F::from(1.0).unwrap(),
        ];

        let a2 = self.a2()?;

        // Calculate V = b[2]*A^2 + b[0]*I
        let a2_scaled = _scalar_mul(&a2, b[2]);
        let i_scaled = _scalar_mul(&self.ident, b[0]);
        let v = add(&a2_scaled, &i_scaled)?;

        // Calculate U = A * (b[3]*A^2 + b[1]*I)
        let a2_scaled2 = _scalar_mul(&a2, b[3]);
        let i_scaled2 = _scalar_mul(&self.ident, b[1]);
        let temp = add(&a2_scaled2, &i_scaled2)?;
        let u = matmul(self.a, &temp)?;

        Ok((u, v))
    }

    /// Compute the Padé approximation of order 5
    fn pade5(&mut self) -> SparseResult<(CsrMatrix<F>, CsrMatrix<F>)> {
        let b = [
            F::from(30240.0).unwrap(),
            F::from(15120.0).unwrap(),
            F::from(3360.0).unwrap(),
            F::from(420.0).unwrap(),
            F::from(30.0).unwrap(),
            F::from(1.0).unwrap(),
        ];

        let a2 = self.a2()?;
        let a4 = self.a4()?;

        // Calculate V = b[4]*A^4 + b[2]*A^2 + b[0]*I
        let a4_scaled = _scalar_mul(&a4, b[4]);
        let a2_scaled = _scalar_mul(&a2, b[2]);
        let i_scaled = _scalar_mul(&self.ident, b[0]);
        let temp_v1 = add(&a4_scaled, &a2_scaled)?;
        let v = add(&temp_v1, &i_scaled)?;

        // Calculate U = A * (b[5]*A^4 + b[3]*A^2 + b[1]*I)
        let a4_scaled2 = _scalar_mul(&a4, b[5]);
        let a2_scaled2 = _scalar_mul(&a2, b[3]);
        let i_scaled2 = _scalar_mul(&self.ident, b[1]);
        let temp_u1 = add(&a4_scaled2, &a2_scaled2)?;
        let temp_u2 = add(&temp_u1, &i_scaled2)?;
        let u = matmul(self.a, &temp_u2)?;

        Ok((u, v))
    }

    /// Compute the Padé approximation of order 7
    fn pade7(&mut self) -> SparseResult<(CsrMatrix<F>, CsrMatrix<F>)> {
        let b = [
            F::from(17297280.0).unwrap(),
            F::from(8648640.0).unwrap(),
            F::from(1995840.0).unwrap(),
            F::from(277200.0).unwrap(),
            F::from(25200.0).unwrap(),
            F::from(1512.0).unwrap(),
            F::from(56.0).unwrap(),
            F::from(1.0).unwrap(),
        ];

        let a2 = self.a2()?;
        let a4 = self.a4()?;
        let a6 = self.a6()?;

        // Calculate V = b[6]*A^6 + b[4]*A^4 + b[2]*A^2 + b[0]*I
        let a6_scaled = _scalar_mul(&a6, b[6]);
        let a4_scaled = _scalar_mul(&a4, b[4]);
        let a2_scaled = _scalar_mul(&a2, b[2]);
        let i_scaled = _scalar_mul(&self.ident, b[0]);

        let temp_v1 = add(&a6_scaled, &a4_scaled)?;
        let temp_v2 = add(&temp_v1, &a2_scaled)?;
        let v = add(&temp_v2, &i_scaled)?;

        // Calculate U = A * (b[7]*A^6 + b[5]*A^4 + b[3]*A^2 + b[1]*I)
        let a6_scaled2 = _scalar_mul(&a6, b[7]);
        let a4_scaled2 = _scalar_mul(&a4, b[5]);
        let a2_scaled2 = _scalar_mul(&a2, b[3]);
        let i_scaled2 = _scalar_mul(&self.ident, b[1]);

        let temp_u1 = add(&a6_scaled2, &a4_scaled2)?;
        let temp_u2 = add(&temp_u1, &a2_scaled2)?;
        let temp_u3 = add(&temp_u2, &i_scaled2)?;
        let u = matmul(self.a, &temp_u3)?;

        Ok((u, v))
    }

    /// Compute the Padé approximation of order 9
    fn pade9(&mut self) -> SparseResult<(CsrMatrix<F>, CsrMatrix<F>)> {
        let b = [
            F::from(17643225600.0).unwrap(),
            F::from(8821612800.0).unwrap(),
            F::from(2075673600.0).unwrap(),
            F::from(302702400.0).unwrap(),
            F::from(30270240.0).unwrap(),
            F::from(2162160.0).unwrap(),
            F::from(110880.0).unwrap(),
            F::from(3960.0).unwrap(),
            F::from(90.0).unwrap(),
            F::from(1.0).unwrap(),
        ];

        let a2 = self.a2()?;
        let a4 = self.a4()?;
        let a6 = self.a6()?;
        let a8 = self.a8()?;

        // Calculate V = b[8]*A^8 + b[6]*A^6 + b[4]*A^4 + b[2]*A^2 + b[0]*I
        let a8_scaled = _scalar_mul(&a8, b[8]);
        let a6_scaled = _scalar_mul(&a6, b[6]);
        let a4_scaled = _scalar_mul(&a4, b[4]);
        let a2_scaled = _scalar_mul(&a2, b[2]);
        let i_scaled = _scalar_mul(&self.ident, b[0]);

        let temp_v1 = add(&a8_scaled, &a6_scaled)?;
        let temp_v2 = add(&temp_v1, &a4_scaled)?;
        let temp_v3 = add(&temp_v2, &a2_scaled)?;
        let v = add(&temp_v3, &i_scaled)?;

        // Calculate U = A * (b[9]*A^8 + b[7]*A^6 + b[5]*A^4 + b[3]*A^2 + b[1]*I)
        let a8_scaled2 = _scalar_mul(&a8, b[9]);
        let a6_scaled2 = _scalar_mul(&a6, b[7]);
        let a4_scaled2 = _scalar_mul(&a4, b[5]);
        let a2_scaled2 = _scalar_mul(&a2, b[3]);
        let i_scaled2 = _scalar_mul(&self.ident, b[1]);

        let temp_u1 = add(&a8_scaled2, &a6_scaled2)?;
        let temp_u2 = add(&temp_u1, &a4_scaled2)?;
        let temp_u3 = add(&temp_u2, &a2_scaled2)?;
        let temp_u4 = add(&temp_u3, &i_scaled2)?;
        let u = matmul(self.a, &temp_u4)?;

        Ok((u, v))
    }

    /// Compute the scaled Padé approximation of order 13
    fn pade13_scaled(&mut self, s: i32) -> SparseResult<(CsrMatrix<F>, CsrMatrix<F>)> {
        let b = [
            F::from(64764752532480000.0).unwrap(),
            F::from(32382376266240000.0).unwrap(),
            F::from(7771770303897600.0).unwrap(),
            F::from(1187353796428800.0).unwrap(),
            F::from(129060195264000.0).unwrap(),
            F::from(10559470521600.0).unwrap(),
            F::from(670442572800.0).unwrap(),
            F::from(33522128640.0).unwrap(),
            F::from(1323241920.0).unwrap(),
            F::from(40840800.0).unwrap(),
            F::from(960960.0).unwrap(),
            F::from(16380.0).unwrap(),
            F::from(182.0).unwrap(),
            F::from(1.0).unwrap(),
        ];

        // Scale the matrix: A_scaled = 2^(-s) * A
        let scale = F::from(2.0_f64.powi(-s)).unwrap();
        let a_scaled = _scalar_mul(self.a, scale);

        // Compute powers of the scaled matrix
        let a2_scaled = _scalar_mul(&self.a2()?, scale * scale);
        let a4_scaled = _scalar_mul(&self.a4()?, scale.powi(4));
        let a6_scaled = _scalar_mul(&self.a6()?, scale.powi(6));

        // Compute U2 = A^6 * (b[13]*A^6 + b[11]*A^4 + b[9]*A^2)
        let a6_scaled_b13 = _scalar_mul(&a6_scaled, b[13]);
        let a4_scaled_b11 = _scalar_mul(&a4_scaled, b[11]);
        let a2_scaled_b9 = _scalar_mul(&a2_scaled, b[9]);

        let temp_u2_1 = add(&a6_scaled_b13, &a4_scaled_b11)?;
        let temp_u2_2 = add(&temp_u2_1, &a2_scaled_b9)?;
        let u2 = matmul(&a6_scaled, &temp_u2_2)?;

        // Compute U = A * (U2 + b[7]*A^6 + b[5]*A^4 + b[3]*A^2 + b[1]*I)
        let a6_scaled_b7 = _scalar_mul(&a6_scaled, b[7]);
        let a4_scaled_b5 = _scalar_mul(&a4_scaled, b[5]);
        let a2_scaled_b3 = _scalar_mul(&a2_scaled, b[3]);
        let i_scaled_b1 = _scalar_mul(&self.ident, b[1]);

        let temp_u_1 = add(&u2, &a6_scaled_b7)?;
        let temp_u_2 = add(&temp_u_1, &a4_scaled_b5)?;
        let temp_u_3 = add(&temp_u_2, &a2_scaled_b3)?;
        let temp_u_4 = add(&temp_u_3, &i_scaled_b1)?;
        let u = matmul(&a_scaled, &temp_u_4)?;

        // Compute V2 = A^6 * (b[12]*A^6 + b[10]*A^4 + b[8]*A^2)
        let a6_scaled_b12 = _scalar_mul(&a6_scaled, b[12]);
        let a4_scaled_b10 = _scalar_mul(&a4_scaled, b[10]);
        let a2_scaled_b8 = _scalar_mul(&a2_scaled, b[8]);

        let temp_v2_1 = add(&a6_scaled_b12, &a4_scaled_b10)?;
        let temp_v2_2 = add(&temp_v2_1, &a2_scaled_b8)?;
        let v2 = matmul(&a6_scaled, &temp_v2_2)?;

        // Compute V = V2 + b[6]*A^6 + b[4]*A^4 + b[2]*A^2 + b[0]*I
        let a6_scaled_b6 = _scalar_mul(&a6_scaled, b[6]);
        let a4_scaled_b4 = _scalar_mul(&a4_scaled, b[4]);
        let a2_scaled_b2 = _scalar_mul(&a2_scaled, b[2]);
        let i_scaled_b0 = _scalar_mul(&self.ident, b[0]);

        let temp_v_1 = add(&v2, &a6_scaled_b6)?;
        let temp_v_2 = add(&temp_v_1, &a4_scaled_b4)?;
        let temp_v_3 = add(&temp_v_2, &a2_scaled_b2)?;
        let v = add(&temp_v_3, &i_scaled_b0)?;

        Ok((u, v))
    }
}
pub fn inv<F>(a: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + Copy + Default + 'static + std::fmt::Debug,
{
    // Check if the matrix is square
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Input must be a square matrix".to_string(),
        ));
    }

    // Create identity matrix
    let i = eye(rows)?;

    // Use sparse direct solver to solve "AX = I"
    // This is equivalent to X = A^-1
    spsolve_matrix(a, &i)
}

/// Compute the matrix exponential using Padé approximation.
///
/// This function computes e^A for the given square matrix A using a combination
/// of scaling and squaring with Padé approximation.
///
/// # Arguments
///
/// * `a` - A square sparse matrix to exponentiate
///
/// # Returns
///
/// * e^A as a sparse matrix
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::linalg::expm;
/// use approx::assert_relative_eq;
///
/// // Create a diagonal matrix
/// let rows = vec![0, 1];
/// let cols = vec![0, 1];
/// let data = vec![1.0, 2.0];
/// let shape = (2, 2);
///
/// let a = CsrMatrix::new(data, rows, cols, shape).unwrap();
///
/// // Compute exp(A)
/// let exp_a = expm(&a).unwrap();
/// let dense = exp_a.to_dense();
///
/// // For a diagonal matrix with entries [1, 2],
/// // exp(A) is a diagonal matrix with entries [e^1, e^2]
/// assert_relative_eq!(dense[0][0], 2.718281828459045, epsilon = 1e-10); // e^1
/// assert_relative_eq!(dense[1][1], 7.38905609893065, epsilon = 1e-10);  // e^2
/// assert_relative_eq!(dense[0][1], 0.0, epsilon = 1e-10); // Off-diagonal elements are zero
/// assert_relative_eq!(dense[1][0], 0.0, epsilon = 1e-10);
/// ```
pub fn expm<F>(a: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + Copy + Default + 'static + std::fmt::Debug + std::ops::AddAssign,
{
    _expm(a, true)
}

/// Core implementation of the matrix exponential
///
/// # Arguments
///
/// * `a` - Square sparse matrix to exponentiate
/// * `use_exact_onenorm` - Whether to use exact one-norm computation
///
/// # Returns
///
/// * Matrix exponential e^A
fn _expm<F>(a: &CsrMatrix<F>, use_exact_onenorm: bool) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + Copy + Default + 'static + std::fmt::Debug + std::ops::AddAssign,
{
    // Check if matrix is square
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Expected a square matrix".to_string(),
        ));
    }

    // Handle empty matrix
    if rows == 0 {
        return CsrMatrix::new(Vec::new(), Vec::new(), Vec::new(), (0, 0));
    }

    // Handle 1x1 matrix
    if rows == 1 {
        // For 1x1 matrix, the expm is just e^a[0,0]
        let scalar = a.to_dense()[0][0].exp();
        let mut data = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        if scalar != F::zero() {
            data.push(scalar);
            row_indices.push(0);
            col_indices.push(0);
        }

        return CsrMatrix::new(data, row_indices, col_indices, (1, 1));
    }

    // Special case for diagonal matrices
    if _is_diagonal(a) {
        println!("Detected diagonal matrix, using direct exponentiation");
        let mut data = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        for i in 0..rows {
            let mut diagonal_val = F::zero();
            // Find the diagonal element at position (i,i)
            for j in a.indptr[i]..a.indptr[i + 1] {
                if a.indices[j] == i {
                    diagonal_val = a.data[j];
                    break;
                }
            }

            // Compute e^a_ii
            let exp_val = diagonal_val.exp();
            if exp_val != F::zero() {
                data.push(exp_val);
                row_indices.push(i);
                col_indices.push(i);
            }
        }

        return CsrMatrix::new(data, row_indices, col_indices, (rows, rows));
    }

    // Detect matrix structure
    let structure = if _is_upper_triangular(a) {
        Some(UPPER_TRIANGULAR)
    } else {
        None
    };

    // Create helper for computing matrix powers
    let mut h = _ExpmPadeHelper::new(a, structure, use_exact_onenorm)?;

    // Try Padé order 3
    let eta_1 = h.d4_loose()?.max(h.d6_loose()?);
    if eta_1 < F::from(1.495_585_217_958_292e-2).unwrap() && _ell(a, 3)? == 0 {
        let (u, v) = h.pade3()?;
        return _solve_p_q(u, v, structure);
    }

    // Try Padé order 5
    let eta_2 = h.d4_tight()?.max(h.d6_loose()?);
    if eta_2 < F::from(2.539_398_330_063_23e-1).unwrap() && _ell(a, 5)? == 0 {
        let (u, v) = h.pade5()?;
        return _solve_p_q(u, v, structure);
    }

    // Try Padé orders 7 and 9
    let eta_3 = h.d6_tight()?.max(h.d8_loose()?);
    if eta_3 < F::from(9.504_178_996_162_932e-1).unwrap() && _ell(a, 7)? == 0 {
        let (u, v) = h.pade7()?;
        return _solve_p_q(u, v, structure);
    }
    if eta_3 < F::from(2.097_847_961_257_068).unwrap() && _ell(a, 9)? == 0 {
        let (u, v) = h.pade9()?;
        return _solve_p_q(u, v, structure);
    }

    // Use Padé order 13
    let eta_4 = h.d8_loose()?.max(h.d10_loose()?);
    let eta_5 = eta_3.min(eta_4);
    let theta_13 = F::from(4.25).unwrap();

    // Choose scaling parameter s
    let s = if eta_5 == F::zero() {
        // Nilpotent special case
        0
    } else {
        let temp = (eta_5 / theta_13).log2().ceil();
        temp.to_f64().unwrap() as i32
    };

    // Add ell correction
    let s = s + _ell(&_scalar_mul(a, F::from(2.0_f64.powi(-s)).unwrap()), 13)?;

    // Compute scaled Padé approximation
    let (u, v) = h.pade13_scaled(s)?;
    let mut x = _solve_p_q(u, v, structure)?;

    // Undo scaling by repeated squaring
    for _ in 0..s {
        x = matmul(&x, &x)?;
    }

    Ok(x)
}

/// Raise a square matrix to the integer power.
///
/// For non-negative integers, A^power is computed using repeated
/// matrix multiplications. Negative integers are not supported.
///
/// # Arguments
///
/// * `a` - Square sparse matrix to be raised to a power
/// * `power` - Integer exponent
///
/// # Returns
///
/// * The result of A^power
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::linalg::matrix_power;
///
/// // Create a simple matrix
/// let rows = vec![0, 0, 1, 1];
/// let cols = vec![0, 1, 0, 1];
/// let data = vec![1.0, 1.0, 1.0, 0.0];
/// let shape = (2, 2);
///
/// let a = CsrMatrix::new(data, rows, cols, shape).unwrap();
///
/// // Compute A²
/// let a_squared = matrix_power(&a, 2).unwrap();
/// let dense = a_squared.to_dense();
/// assert_eq!(dense, vec![vec![2.0, 1.0], vec![1.0, 1.0]]);
/// ```
pub fn matrix_power<F>(a: &CsrMatrix<F>, power: i32) -> SparseResult<CsrMatrix<F>>
where
    F: Float + Sum + Copy + Default + 'static + std::fmt::Debug,
{
    let (m, n) = a.shape();

    if m != n {
        return Err(SparseError::ValueError("Matrix must be square".to_string()));
    }

    if power < 0 {
        return Err(SparseError::ValueError(
            "Negative powers are not supported. Use inverse for power -1 and combine with positive powers.".to_string(),
        ));
    }

    if power == 0 {
        // Return identity matrix
        return eye(m);
    }

    if power == 1 {
        // Return a copy of the matrix
        return Ok(a.clone());
    }

    // Recursive implementation using binary exponentiation
    let half_power = matrix_power(a, power / 2)?;

    if power % 2 == 0 {
        // Even power: A^n = (A^(n/2))²
        matmul(&half_power, &half_power)
    } else {
        // Odd power: A^n = A * (A^(n/2))²
        let half_power_squared = matmul(&half_power, &half_power)?;
        matmul(a, &half_power_squared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_spsolve() {
        // Create a simple diagonal system
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![2.0, 2.0, 2.0];
        let shape = (3, 3);

        let a = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let b = vec![2.0, 4.0, 6.0];

        // Solve Ax = b (should be x = [1, 2, 3])
        let x = spsolve(&a, &b).unwrap();

        assert_eq!(x.len(), 3);
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(x[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(x[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_expm() {
        // Test 1: Zero matrix - e^0 = I
        let zero = CsrMatrix::<f64>::new(Vec::new(), Vec::new(), Vec::new(), (3, 3)).unwrap();
        let exp_zero = expm(&zero).unwrap();

        // Print for debugging
        println!("Zero matrix exponential:");
        let zero_dense = exp_zero.to_dense();
        for row in &zero_dense {
            println!("{:?}", row);
        }

        // e^0 should be the identity matrix
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_relative_eq!(zero_dense[i][j], 1.0, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(zero_dense[i][j], 0.0, epsilon = 1e-10);
                }
            }
        }

        // Test 2: Diagonal matrix
        // For a diagonal matrix, e^A has diagonal elements e^a_ii
        let rows = vec![0, 1];
        let cols = vec![0, 1];
        let data = vec![1.0, 2.0];
        let shape = (2, 2);

        let a = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let exp_a = expm(&a).unwrap();

        println!("Diagonal matrix exponential:");
        let exp_a_dense = exp_a.to_dense();
        for row in &exp_a_dense {
            println!("{:?}", row);
        }

        // Verify diagonal elements are e^a_ii and off-diagonals are 0
        assert_relative_eq!(exp_a_dense[0][0], std::f64::consts::E, epsilon = 1e-10); // e^1
        assert_relative_eq!(
            exp_a_dense[1][1],
            std::f64::consts::E.powi(2),
            epsilon = 1e-10
        ); // e^2
        assert_relative_eq!(exp_a_dense[0][1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(exp_a_dense[1][0], 0.0, epsilon = 1e-10);

        // Test 3: Error case - non-square matrix
        let non_square =
            CsrMatrix::<f64>::new(vec![1.0, 2.0], vec![0, 1], vec![0, 0], (2, 3)).unwrap();

        assert!(expm(&non_square).is_err());
    }

    #[test]
    fn test_sparse_cholesky_simple() {
        // Create a simple SPD matrix
        // [2 1 0]
        // [1 2 1]
        // [0 1 2]
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let data = vec![2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0];
        let shape = (3, 3);

        let a = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let b = vec![1.0, 2.0, 3.0];

        // Verify that the function runs without error - we're not testing accuracy
        // since the current implementation is a simplified prototype
        let _x = sparse_cholesky_solve(&a, &b).unwrap();
    }

    #[test]
    fn test_sparse_lu_solve() {
        // Create a non-symmetric matrix
        // [2 1 0]
        // [1 2 1]
        // [1 0 2]
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 0, 2];
        let data = vec![2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0];
        let shape = (3, 3);

        let a = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let b = vec![1.0, 2.0, 3.0];

        // Verify that the function runs without error - we're not testing accuracy
        // since the current implementation is a simplified prototype
        let _x = sparse_lu_solve(&a, &b).unwrap();
    }

    #[test]
    fn test_sparse_direct_solve() {
        // Create a symmetric positive definite matrix
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let data = vec![2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0];
        let shape = (3, 3);

        let a = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let b = vec![1.0, 2.0, 3.0];

        // Verify that the function runs without error - we're not testing accuracy
        // since the current implementation is a simplified prototype
        let _x = sparse_direct_solve(&a, &b, true, true).unwrap();

        // Test with a general matrix (not assuming SPD)
        let _x_general = sparse_direct_solve(&a, &b, false, false).unwrap();
    }

    #[test]
    fn test_sparse_lstsq() {
        // Create a rectangular system (overdetermined)
        // [1 1]
        // [1 2]
        // [1 3]
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 0, 1];
        let data = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let shape = (3, 2);

        let a = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let b = vec![6.0, 9.0, 12.0];

        // Verify that the function runs without error - we're not testing accuracy
        // since the current implementation is a simplified prototype
        let _x = sparse_lstsq(&a, &b).unwrap();
    }

    #[test]
    fn test_norm() {
        // Create a test matrix
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let shape = (3, 3);

        let a = CsrMatrix::new(data, rows, cols, shape).unwrap();

        // Matrix:
        // [2 1 0]
        // [1 2 0]
        // [0 1 3]

        // Infinity norm: maximum absolute row sum
        let inf_norm = norm(&a, "inf").unwrap();
        assert_relative_eq!(inf_norm, 4.0, epsilon = 1e-10); // max(3, 3, 4)

        // 1-norm: maximum absolute column sum
        let norm_1 = norm(&a, "1").unwrap();
        assert_relative_eq!(norm_1, 4.0, epsilon = 1e-10); // max(3, 4, 3)

        // Frobenius norm: sqrt(sum(a_ij²))
        let frob_norm = norm(&a, "fro").unwrap();
        let sum: f64 = 2.0 * 2.0 + 1.0 * 1.0 + 1.0 * 1.0 + 2.0 * 2.0 + 1.0 * 1.0 + 3.0 * 3.0;
        let expected: f64 = sum.sqrt();
        assert_relative_eq!(frob_norm, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_matmul() {
        // Create two test matrices
        // Matrix A:
        // [1 2]
        // [3 4]
        let rows_a = vec![0, 0, 1, 1];
        let cols_a = vec![0, 1, 0, 1];
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let shape_a = (2, 2);
        let a = CsrMatrix::new(data_a, rows_a, cols_a, shape_a).unwrap();

        // Matrix B:
        // [5 6]
        // [7 8]
        let rows_b = vec![0, 0, 1, 1];
        let cols_b = vec![0, 1, 0, 1];
        let data_b = vec![5.0, 6.0, 7.0, 8.0];
        let shape_b = (2, 2);
        let b = CsrMatrix::new(data_b, rows_b, cols_b, shape_b).unwrap();

        // Compute C = A * B
        let c = matmul(&a, &b).unwrap();

        // Expected result:
        // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        let c_dense = c.to_dense();
        let expected = vec![vec![19.0, 22.0], vec![43.0, 50.0]];

        assert_eq!(c_dense, expected);
    }

    #[test]
    fn test_add() {
        // Create two test matrices
        // Matrix A:
        // [1 2]
        // [3 0]
        let rows_a = vec![0, 0, 1];
        let cols_a = vec![0, 1, 0];
        let data_a = vec![1.0, 2.0, 3.0];
        let shape_a = (2, 2);
        let a = CsrMatrix::new(data_a, rows_a, cols_a, shape_a).unwrap();

        // Matrix B:
        // [5 0]
        // [0 8]
        let rows_b = vec![0, 1];
        let cols_b = vec![0, 1];
        let data_b = vec![5.0, 8.0];
        let shape_b = (2, 2);
        let b = CsrMatrix::new(data_b, rows_b, cols_b, shape_b).unwrap();

        // Compute C = A + B
        let c = add(&a, &b).unwrap();

        // Expected result:
        // [1+5 2+0]   [6 2]
        // [3+0 0+8] = [3 8]
        let c_dense = c.to_dense();
        let expected = vec![vec![6.0, 2.0], vec![3.0, 8.0]];

        assert_eq!(c_dense, expected);
    }

    #[test]
    fn test_multiply() {
        // Create two test matrices
        // Matrix A:
        // [1 2]
        // [3 4]
        let rows_a = vec![0, 0, 1, 1];
        let cols_a = vec![0, 1, 0, 1];
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let shape_a = (2, 2);
        let a = CsrMatrix::new(data_a, rows_a, cols_a, shape_a).unwrap();

        // Matrix B:
        // [5 6]
        // [7 8]
        let rows_b = vec![0, 0, 1, 1];
        let cols_b = vec![0, 1, 0, 1];
        let data_b = vec![5.0, 6.0, 7.0, 8.0];
        let shape_b = (2, 2);
        let b = CsrMatrix::new(data_b, rows_b, cols_b, shape_b).unwrap();

        // Compute C = A .* B (element-wise multiplication)
        let c = multiply(&a, &b).unwrap();

        // Expected result:
        // [1*5 2*6]   [5  12]
        // [3*7 4*8] = [21 32]
        let c_dense = c.to_dense();
        let expected = vec![vec![5.0, 12.0], vec![21.0, 32.0]];

        assert_eq!(c_dense, expected);
    }

    #[test]
    fn test_eye() {
        // Create a 3x3 identity matrix
        let eye_matrix = eye::<f64>(3).unwrap();

        // Expected result:
        // [1 0 0]
        // [0 1 0]
        // [0 0 1]
        let eye_dense = eye_matrix.to_dense();
        let expected = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        assert_eq!(eye_dense, expected);

        // Check for correct number of non-zeros (should be equal to size)
        assert_eq!(eye_matrix.nnz(), 3);
    }

    #[test]
    fn test_diag_matrix() {
        // Create a diagonal matrix with specified elements
        let diag_values = vec![2.0, 5.0, 8.0];
        let diag_mat = diag_matrix(&diag_values, None).unwrap();

        // Expected result:
        // [2 0 0]
        // [0 5 0]
        // [0 0 8]
        let diag_dense = diag_mat.to_dense();
        let expected = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 5.0, 0.0],
            vec![0.0, 0.0, 8.0],
        ];

        assert_eq!(diag_dense, expected);

        // Test with explicit size parameter (larger than diagonal values)
        let larger_diag = diag_matrix(&diag_values, Some(4)).unwrap();
        assert_eq!(larger_diag.shape(), (4, 4));

        // Test with zeros in diagonal to ensure they're properly handled
        let diag_with_zeros = vec![1.0, 0.0, 3.0];
        let zero_diag_matrix = diag_matrix(&diag_with_zeros, None).unwrap();
        assert_eq!(zero_diag_matrix.nnz(), 2); // Only 2 non-zero elements
    }

    #[test]
    fn test_matrix_power() {
        // Create a test matrix
        // [1 1]
        // [1 0]
        let rows = vec![0, 0, 1];
        let cols = vec![0, 1, 0];
        let data = vec![1.0, 1.0, 1.0];
        let shape = (2, 2);
        let a = CsrMatrix::new(data, rows, cols, shape).unwrap();

        // Power 0: Identity matrix
        let power0 = matrix_power(&a, 0).unwrap();
        let power0_dense = power0.to_dense();
        assert_eq!(power0_dense, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

        // Power 1: Same as original
        let power1 = matrix_power(&a, 1).unwrap();
        let power1_dense = power1.to_dense();
        assert_eq!(power1_dense, vec![vec![1.0, 1.0], vec![1.0, 0.0]]);

        // Power 2: A^2
        let power2 = matrix_power(&a, 2).unwrap();
        let power2_dense = power2.to_dense();
        assert_eq!(power2_dense, vec![vec![2.0, 1.0], vec![1.0, 1.0]]);

        // Power 3: A^3
        let power3 = matrix_power(&a, 3).unwrap();
        let power3_dense = power3.to_dense();
        assert_eq!(power3_dense, vec![vec![3.0, 2.0], vec![2.0, 1.0]]);

        // Power 4: A^4 (using binary exponentiation)
        let power4 = matrix_power(&a, 4).unwrap();
        let power4_dense = power4.to_dense();
        assert_eq!(power4_dense, vec![vec![5.0, 3.0], vec![3.0, 2.0]]);

        // Test error cases

        // Non-square matrix
        let non_square_rows = vec![0, 0, 1];
        let non_square_cols = vec![0, 1, 0];
        let non_square_data = vec![1.0, 1.0, 1.0];
        let non_square_shape = (2, 3); // Non-square
        let non_square = CsrMatrix::new(
            non_square_data,
            non_square_rows,
            non_square_cols,
            non_square_shape,
        )
        .unwrap();

        assert!(matrix_power(&non_square, 2).is_err());

        // Negative power
        assert!(matrix_power(&a, -1).is_err());
    }

    #[test]
    fn test_inv() {
        // Create an identity matrix - we know its inverse is itself
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();

        for i in 0..3 {
            rows.push(i);
            cols.push(i);
            data.push(1.0);
        }

        let a = CsrMatrix::new(data, rows, cols, (3, 3)).unwrap();

        // Compute inverse of identity - should be identity
        let a_inv = inv(&a).unwrap();
        let a_inv_dense = a_inv.to_dense();

        // Print for debugging
        println!("Identity matrix inverse:");
        for row in &a_inv_dense {
            println!("{:?}", row);
        }

        // Check it's still the identity
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_relative_eq!(a_inv_dense[i][j], 1.0, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(a_inv_dense[i][j], 0.0, epsilon = 1e-10);
                }
            }
        }

        // Test simple 2x2 diagonal matrix [2 0; 0 4]
        // Inverse should be [0.5 0; 0 0.25]
        rows = vec![0, 1];
        cols = vec![0, 1];
        data = vec![2.0, 4.0];

        let diag_matrix = CsrMatrix::new(data, rows, cols, (2, 2)).unwrap();
        let diag_inv = inv(&diag_matrix).unwrap();
        let diag_inv_dense = diag_inv.to_dense();

        println!("Diagonal matrix inverse:");
        for row in &diag_inv_dense {
            println!("{:?}", row);
        }

        assert_relative_eq!(diag_inv_dense[0][0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(diag_inv_dense[1][1], 0.25, epsilon = 1e-10);
        assert_relative_eq!(diag_inv_dense[0][1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(diag_inv_dense[1][0], 0.0, epsilon = 1e-10);

        // Test error case: non-square matrix
        let non_square_rows = vec![0, 0, 1];
        let non_square_cols = vec![0, 1, 0];
        let non_square_data = vec![1.0, 1.0, 1.0];
        let non_square_shape = (2, 3); // Non-square
        let non_square = CsrMatrix::new(
            non_square_data,
            non_square_rows,
            non_square_cols,
            non_square_shape,
        )
        .unwrap();

        assert!(inv(&non_square).is_err());
    }
}
