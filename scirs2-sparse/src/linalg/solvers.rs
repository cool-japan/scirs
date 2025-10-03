//! Direct solvers and basic operations for sparse matrices

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

// Re-export the functions from the original linalg.rs
// For now, we'll implement these functions here. In a real migration,
// we would move the implementations from linalg.rs to here.

// I'll implement stubs for the functions that need to be moved.
// The actual implementations should be copied from linalg.rs

/// Solve a sparse linear system Ax = b
#[allow(dead_code)]
pub fn spsolve<F>(a: &CsrMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    // This implementation should be moved from linalg.rs
    // For now, I'll forward to sparse_direct_solve
    // For now, use a simple Gaussian elimination approach
    let a_dense = a.to_dense();
    gaussian_elimination(&a_dense, b)
}

/// Solve a sparse linear system using direct methods
#[allow(dead_code)]
pub fn sparse_direct_solve<F>(
    a: &CsrMatrix<F>,
    b: &[F],
    _symmetric: bool,
    _positive_definite: bool,
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

    // For this stub implementation, we'll use Gaussian elimination
    // The real implementation should use optimized sparse solvers
    let a_dense = a.to_dense();
    gaussian_elimination(&a_dense, b)
}

/// Solve a least squares problem
#[allow(dead_code)]
pub fn sparse_lstsq<F>(a: &CsrMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    // For now, solve normal equations: A^T * A * x = A^T * b
    let at = a.transpose();
    let ata = matmul(&at, a)?;
    // Compute A^T * b
    let mut atb = vec![F::zero(); at.rows()];
    for (row, atb_val) in atb.iter_mut().enumerate().take(at.rows()) {
        let row_range = at.row_range(row);
        let row_indices = &at.indices[row_range.clone()];
        let row_data = &at.data[row_range];

        let mut sum = F::zero();
        for (col_idx, &col) in row_indices.iter().enumerate() {
            sum += row_data[col_idx] * b[col];
        }
        *atb_val = sum;
    }
    spsolve(&ata, &atb)
}

/// Compute matrix norm
#[allow(dead_code)]
pub fn norm<F>(a: &CsrMatrix<F>, ord: &str) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    match ord {
        "1" => {
            // 1-norm: maximum column sum
            let mut max_sum = F::zero();
            for j in 0..a.cols() {
                let mut col_sum = F::zero();
                for i in 0..a.rows() {
                    let val = a.get(i, j);
                    if val != F::zero() {
                        col_sum += val.abs();
                    }
                }
                if col_sum > max_sum {
                    max_sum = col_sum;
                }
            }
            Ok(max_sum)
        }
        "inf" => {
            // Infinity norm: maximum row sum
            let mut max_sum = F::zero();
            for i in 0..a.rows() {
                let mut row_sum = F::zero();
                for j in 0..a.cols() {
                    let val = a.get(i, j);
                    if val != F::zero() {
                        row_sum += val.abs();
                    }
                }
                if row_sum > max_sum {
                    max_sum = row_sum;
                }
            }
            Ok(max_sum)
        }
        "fro" => {
            // Frobenius norm: sqrt(sum of squares)
            let sum_squares: F = a.data.iter().map(|v| *v * *v).sum();
            Ok(sum_squares.sqrt())
        }
        _ => Err(SparseError::ValueError(format!("Unknown norm: {ord}"))),
    }
}

/// Matrix multiplication
#[allow(dead_code)]
pub fn matmul<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    // Matrix multiplication - use a simple implementation
    let mut result_rows = Vec::new();
    let mut result_cols = Vec::new();
    let mut result_data = Vec::new();

    for i in 0..a.rows() {
        for j in 0..b.cols() {
            let mut sum = F::zero();
            for k in 0..a.cols() {
                sum += a.get(i, k) * b.get(k, j);
            }
            if sum != F::zero() {
                result_rows.push(i);
                result_cols.push(j);
                result_data.push(sum);
            }
        }
    }

    CsrMatrix::new(result_data, result_rows, result_cols, (a.rows(), b.cols()))
}

/// Matrix addition
#[allow(dead_code)]
pub fn add<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    if a.shape() != b.shape() {
        return Err(SparseError::ShapeMismatch {
            expected: a.shape(),
            found: b.shape(),
        });
    }

    // Simple implementation using dense matrices
    let a_dense = a.to_dense();
    let b_dense = b.to_dense();

    let mut result_dense = vec![vec![F::zero(); a.cols()]; a.rows()];
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            result_dense[i][j] = a_dense[i][j] + b_dense[i][j];
        }
    }

    // Convert back to CSR
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for (i, row) in result_dense.iter().enumerate().take(a.rows()) {
        for (j, &val) in row.iter().enumerate().take(a.cols()) {
            if val != F::zero() {
                rows.push(i);
                cols.push(j);
                data.push(val);
            }
        }
    }

    CsrMatrix::new(data, rows, cols, a.shape())
}

/// Element-wise multiplication (Hadamard product)
#[allow(dead_code)]
pub fn multiply<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    if a.shape() != b.shape() {
        return Err(SparseError::ShapeMismatch {
            expected: a.shape(),
            found: b.shape(),
        });
    }

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    // Only multiply where both matrices have non-zero entries
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            let a_val = a.get(i, j);
            let b_val = b.get(i, j);
            if a_val != F::zero() && b_val != F::zero() {
                rows.push(i);
                cols.push(j);
                data.push(a_val * b_val);
            }
        }
    }

    CsrMatrix::new(data, rows, cols, a.shape())
}

/// Create a diagonal matrix
#[allow(dead_code)]
pub fn diag_matrix<F>(diag: &[F], n: Option<usize>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    let size = n.unwrap_or(diag.len());
    if size < diag.len() {
        return Err(SparseError::ValueError(
            "Size must be at least as large as diagonal".to_string(),
        ));
    }

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for (i, &val) in diag.iter().enumerate() {
        if val != F::zero() {
            rows.push(i);
            cols.push(i);
            data.push(val);
        }
    }

    CsrMatrix::new(data, rows, cols, (size, size))
}

/// Create an identity matrix
#[allow(dead_code)]
pub fn eye<F>(n: usize) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    let diag = vec![F::one(); n];
    diag_matrix(&diag, Some(n))
}

/// Matrix inverse
#[allow(dead_code)]
pub fn inv<F>(a: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    if a.rows() != a.cols() {
        return Err(SparseError::ValueError(
            "Matrix must be square for inverse".to_string(),
        ));
    }

    let n = a.rows();

    // Solve A * X = I for X
    let mut inv_cols = Vec::new();

    for j in 0..n {
        // Get column j from identity matrix
        let mut col_vec = vec![F::zero(); n];
        col_vec[j] = F::one();
        let x = spsolve(a, &col_vec)?;
        inv_cols.push(x);
    }

    // Construct the inverse matrix from columns
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for (j, col) in inv_cols.iter().enumerate() {
        for (i, &val) in col.iter().enumerate() {
            if val.abs() > F::epsilon() {
                rows.push(i);
                cols.push(j);
                data.push(val);
            }
        }
    }

    CsrMatrix::new(data, rows, cols, (n, n))
}

// Matrix exponential functionality is now available in linalg/expm.rs module

/// Matrix power
#[allow(dead_code)]
pub fn matrix_power<F>(a: &CsrMatrix<F>, power: i32) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + 'static + std::fmt::Debug,
{
    if a.rows() != a.cols() {
        return Err(SparseError::ValueError(
            "Matrix must be square for power".to_string(),
        ));
    }

    match power {
        0 => eye(a.rows()),
        1 => Ok(a.clone()),
        p if p > 0 => {
            let mut result = a.clone();
            for _ in 1..p {
                result = matmul(&result, a)?;
            }
            Ok(result)
        }
        p => {
            // Negative power: compute inverse and then positive power
            let inv_a = inv(a)?;
            matrix_power(&inv_a, -p)
        }
    }
}

// Helper functions

#[allow(dead_code)]
fn gaussian_elimination<F>(a: &[Vec<F>], b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign,
{
    let n = a.len();
    let mut aug = vec![vec![F::zero(); n + 1]; n];

    // Create augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        for i in (k + 1)..n {
            if aug[i][k].abs() > aug[max_idx][k].abs() {
                max_idx = i;
            }
        }
        aug.swap(k, max_idx);

        // Check for zero pivot
        if aug[k][k].abs() < F::epsilon() {
            return Err(SparseError::SingularMatrix(
                "Matrix is singular".to_string(),
            ));
        }

        // Eliminate column
        for i in (k + 1)..n {
            let factor = aug[i][k] / aug[k][k];
            for j in k..=n {
                aug[i][j] = aug[i][j] - factor * aug[k][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![F::zero(); n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] = x[i] - aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Ok(x)
}

// Helper functions for matrix exponential have been moved to linalg/expm.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eye_matrix() {
        let eye_matrix = eye::<f64>(3).unwrap();
        assert_eq!(eye_matrix.shape(), (3, 3));
        assert_eq!(eye_matrix.get(0, 0), 1.0);
        assert_eq!(eye_matrix.get(1, 1), 1.0);
        assert_eq!(eye_matrix.get(2, 2), 1.0);
        assert_eq!(eye_matrix.get(0, 1), 0.0);
    }

    #[test]
    fn test_diag_matrix() {
        let diag = vec![2.0, 3.0, 4.0];
        let diag_matrix = diag_matrix(&diag, None).unwrap();
        assert_eq!(diag_matrix.shape(), (3, 3));
        assert_eq!(diag_matrix.get(0, 0), 2.0);
        assert_eq!(diag_matrix.get(1, 1), 3.0);
        assert_eq!(diag_matrix.get(2, 2), 4.0);
    }

    #[test]
    fn test_matrix_power() {
        let diag = vec![2.0, 3.0];
        let matrix = diag_matrix(&diag, None).unwrap();

        // Test power 2
        let matrix2 = matrix_power(&matrix, 2).unwrap();
        assert_eq!(matrix2.get(0, 0), 4.0);
        assert_eq!(matrix2.get(1, 1), 9.0);

        // Test power 0 (identity)
        let matrix0 = matrix_power(&matrix, 0).unwrap();
        assert_eq!(matrix0.get(0, 0), 1.0);
        assert_eq!(matrix0.get(1, 1), 1.0);
    }
}
