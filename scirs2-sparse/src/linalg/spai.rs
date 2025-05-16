//! Sparse Approximate Inverse (SPAI) preconditioner

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use crate::linalg::interface::LinearOperator;
use num_traits::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

/// Sparse Approximate Inverse (SPAI) preconditioner
///
/// This preconditioner computes a sparse approximate inverse M ≈ A^(-1)
/// using a minimization approach: minimize ||I - AM||_F subject to
/// sparsity constraints on M.
pub struct SpaiPreconditioner<F> {
    /// Sparse approximate inverse of the original matrix
    approx_inverse: CsrMatrix<F>,
}

/// Options for the SPAI preconditioner
pub struct SpaiOptions {
    /// Maximum number of nonzeros per column of M
    pub max_nnz_per_col: usize,
    /// Tolerance for least squares solver
    pub ls_tolerance: f64,
    /// Maximum iterations for least squares solver
    pub max_ls_iters: usize,
}

impl Default for SpaiOptions {
    fn default() -> Self {
        Self {
            max_nnz_per_col: 10,
            ls_tolerance: 1e-10,
            max_ls_iters: 100,
        }
    }
}

impl<F: Float + NumAssign + Sum + Debug + 'static> SpaiPreconditioner<F> {
    /// Create a new SPAI preconditioner from a sparse matrix
    pub fn new(matrix: &CsrMatrix<F>, options: SpaiOptions) -> SparseResult<Self> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: matrix.cols(),
            });
        }

        // For now, we'll implement a simplified version of SPAI
        // that uses a static sparsity pattern (diagonal + few off-diagonals)

        // Initialize M as identity matrix in dense format
        let mut m_dense = vec![vec![F::zero(); n]; n];
        for i in 0..n {
            m_dense[i][i] = F::one();
        }

        // For each column of M, solve a least squares problem
        for j in 0..n {
            // Define sparsity pattern for column j
            // For simplicity, we'll use a pattern that includes the diagonal
            // and a few neighboring elements
            let mut pattern = vec![j];

            // Add neighbors within distance 2
            let start = if j >= 2 { j - 2 } else { 0 };
            let end = (j + 3).min(n);

            for k in start..end {
                if k != j && pattern.len() < options.max_nnz_per_col {
                    pattern.push(k);
                }
            }

            // Extract the relevant submatrix A_k from A
            let k = pattern.len();
            let mut a_k = vec![vec![F::zero(); k]; n];

            for (col_idx, &col) in pattern.iter().enumerate() {
                for row in 0..n {
                    let val = matrix.get(row, col);
                    a_k[row][col_idx] = val;
                }
            }

            // Set up right-hand side (j-th unit vector)
            let mut e_j = vec![F::zero(); n];
            e_j[j] = F::one();

            // Solve least squares problem: minimize ||A_k * m_k - e_j||
            // For now, use a simple normal equations approach
            // A_k^T * A_k * m_k = A_k^T * e_j

            // Compute A_k^T * A_k
            let mut ata = vec![vec![F::zero(); k]; k];
            for i in 0..k {
                for j_inner in 0..k {
                    let mut sum = F::zero();
                    for m in 0..n {
                        sum += a_k[m][i] * a_k[m][j_inner];
                    }
                    ata[i][j_inner] = sum;
                }
            }

            // Compute A_k^T * e_j
            let mut atb = vec![F::zero(); k];
            atb[..k].copy_from_slice(&a_k[j][..k]);

            // Solve the system (simplified - in practice, use proper solver)
            let m_k = solve_dense_system(&ata, &atb)?;

            // Update M with the computed values
            for (idx, &row) in pattern.iter().enumerate() {
                m_dense[row][j] = m_k[idx];
            }
        }

        // Convert dense M to sparse format manually
        let n = m_dense.len();
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for i in 0..n {
            for j in 0..n {
                if m_dense[i][j].abs() > F::epsilon() {
                    data.push(m_dense[i][j]);
                    indices.push(j);
                }
            }
            indptr.push(data.len());
        }

        let approx_inverse = CsrMatrix::from_raw_csr(data, indptr, indices, (n, n))?;

        Ok(Self { approx_inverse })
    }
}

impl<F: Float + NumAssign + Sum + Debug + 'static> LinearOperator<F> for SpaiPreconditioner<F> {
    fn shape(&self) -> (usize, usize) {
        self.approx_inverse.shape()
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.approx_inverse.cols() {
            return Err(SparseError::DimensionMismatch {
                expected: self.approx_inverse.cols(),
                found: x.len(),
            });
        }

        let mut result = vec![F::zero(); self.approx_inverse.rows()];

        for (row_idx, result_val) in result.iter_mut().enumerate() {
            for j in self.approx_inverse.indptr[row_idx]..self.approx_inverse.indptr[row_idx + 1] {
                let col_idx = self.approx_inverse.indices[j];
                *result_val += self.approx_inverse.data[j] * x[col_idx];
            }
        }

        Ok(result)
    }

    fn has_adjoint(&self) -> bool {
        true
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // For transpose multiplication, we implement A^T * x
        if x.len() != self.approx_inverse.rows() {
            return Err(SparseError::DimensionMismatch {
                expected: self.approx_inverse.rows(),
                found: x.len(),
            });
        }

        let mut result = vec![F::zero(); self.approx_inverse.cols()];

        for (row_idx, &x_val) in x.iter().enumerate() {
            for j in self.approx_inverse.indptr[row_idx]..self.approx_inverse.indptr[row_idx + 1] {
                let col_idx = self.approx_inverse.indices[j];
                result[col_idx] += self.approx_inverse.data[j] * x_val;
            }
        }

        Ok(result)
    }
}

/// Solve a dense linear system using Gaussian elimination with partial pivoting
fn solve_dense_system<F: Float + NumAssign>(a: &[Vec<F>], b: &[F]) -> SparseResult<Vec<F>> {
    let n = a.len();
    if n == 0 || n != a[0].len() || n != b.len() {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }

    // Create augmented matrix
    let mut aug = vec![vec![F::zero(); n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Gaussian elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = aug[k][k].abs();
        for i in (k + 1)..n {
            if aug[i][k].abs() > max_val {
                max_val = aug[i][k].abs();
                max_row = i;
            }
        }

        // Check for singularity
        if max_val < F::from(1e-14).unwrap() {
            return Err(SparseError::ValueError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows
        if max_row != k {
            aug.swap(k, max_row);
        }

        // Eliminate below
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;

    #[test]
    fn test_spai_simple() {
        // Test with a simple tridiagonal matrix
        // A = [4  -1   0]
        //     [-1  4  -1]
        //     [0  -1   4]
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let indptr = vec![0, 2, 5, 7];
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let matrix = CsrMatrix::from_raw_csr(data, indptr, indices, (3, 3)).unwrap();

        let options = SpaiOptions::default();
        let preconditioner = SpaiPreconditioner::new(&matrix, options).unwrap();

        // Test by applying preconditioner to a vector
        let b = vec![1.0, 2.0, 3.0];
        let x = preconditioner.matvec(&b).unwrap();

        // The result should be approximately the solution to Ax = b
        // For this simple case, we can verify the result is reasonable
        assert!(x.iter().all(|&xi| xi.is_finite()));
    }

    #[test]
    fn test_spai_diagonal() {
        // Test with a diagonal matrix (should get exact inverse)
        // A = [2   0   0]
        //     [0   3   0]
        //     [0   0   4]
        let data = vec![2.0, 3.0, 4.0];
        let indptr = vec![0, 1, 2, 3];
        let indices = vec![0, 1, 2];
        let matrix = CsrMatrix::from_raw_csr(data, indptr, indices, (3, 3)).unwrap();

        let options = SpaiOptions::default();
        let preconditioner = SpaiPreconditioner::new(&matrix, options).unwrap();

        // Apply preconditioner to each unit vector
        let e1 = vec![1.0, 0.0, 0.0];
        let e2 = vec![0.0, 1.0, 0.0];
        let e3 = vec![0.0, 0.0, 1.0];

        let x1 = preconditioner.matvec(&e1).unwrap();
        let x2 = preconditioner.matvec(&e2).unwrap();
        let x3 = preconditioner.matvec(&e3).unwrap();

        // For a diagonal matrix, SPAI should recover the exact inverse
        assert!((x1[0] - 0.5).abs() < 1e-10);
        assert!((x2[1] - 1.0 / 3.0).abs() < 1e-10);
        assert!((x3[2] - 0.25).abs() < 1e-10);
    }
}
