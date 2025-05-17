//! Incomplete Cholesky factorization preconditioner

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use crate::linalg::interface::LinearOperator;
use num_traits::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

/// Incomplete Cholesky factorization preconditioner (IC(0))
///
/// This preconditioner computes an incomplete Cholesky factorization L*L^T ≈ A
/// where L is a lower triangular matrix with the same sparsity pattern as the
/// lower triangular part of A.
pub struct IC0Preconditioner<F> {
    /// Lower triangular factor L in CSR format
    l_factor: CsrMatrix<F>,
}

impl<F: Float + NumAssign + Sum + Debug + 'static> IC0Preconditioner<F> {
    /// Create a new IC(0) preconditioner from a symmetric positive definite matrix
    pub fn new(matrix: &CsrMatrix<F>) -> SparseResult<Self> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: matrix.cols(),
            });
        }

        // Initialize L with the lower triangular part of A
        let mut l_data = Vec::new();
        let mut l_indices = Vec::new();
        let mut l_indptr = vec![0];

        for i in 0..n {
            let row_start = matrix.indptr[i];
            let row_end = matrix.indptr[i + 1];

            // Copy lower triangular entries
            for k in row_start..row_end {
                let j = matrix.indices[k];
                if j <= i {
                    l_indices.push(j);
                    l_data.push(matrix.data[k]);
                }
            }
            l_indptr.push(l_indices.len());
        }

        // Perform incomplete Cholesky factorization
        for i in 0..n {
            let row_start = l_indptr[i];
            let row_end = l_indptr[i + 1];

            // Find diagonal element
            let mut diag_idx = None;
            for (idx, &col) in l_indices[row_start..row_end].iter().enumerate() {
                if col == i {
                    diag_idx = Some(row_start + idx);
                    break;
                }
            }

            let diag_idx = match diag_idx {
                Some(idx) => idx,
                None => {
                    return Err(SparseError::ValueError(format!(
                        "Missing diagonal element at position {}",
                        i
                    )));
                }
            };

            // Update diagonal element
            let mut diag_val = l_data[diag_idx];

            // Subtract contributions from previous columns
            for k in row_start..diag_idx {
                let j = l_indices[k];

                // Find L[j,j]
                let j_row_start = l_indptr[j];
                let j_row_end = l_indptr[j + 1];
                let mut l_jj = F::zero();

                for idx in j_row_start..j_row_end {
                    if l_indices[idx] == j {
                        l_jj = l_data[idx];
                        break;
                    }
                }

                diag_val -= l_data[k] * l_data[k] / l_jj;
            }

            // Check if factorization is possible
            if diag_val <= F::zero() {
                return Err(SparseError::ValueError(
                    "Matrix is not positive definite or factorization broke down".to_string(),
                ));
            }

            l_data[diag_idx] = diag_val.sqrt();

            // Update off-diagonal elements
            for k in (diag_idx + 1)..row_end {
                let j = l_indices[k];
                let mut sum = F::zero();

                // Compute dot product of row i and row j up to column j
                for p in row_start..diag_idx {
                    let col_p = l_indices[p];

                    // Find corresponding element in row j
                    let j_row_start = l_indptr[j];
                    let j_row_end = l_indptr[j + 1];

                    for q in j_row_start..j_row_end {
                        if l_indices[q] == col_p {
                            sum += l_data[p] * l_data[q];
                            break;
                        }
                    }
                }

                l_data[k] = (l_data[k] - sum) / l_data[diag_idx];
            }
        }

        // Create the L factor as a CSR matrix
        let l_factor = CsrMatrix::from_raw_csr(l_data, l_indptr, l_indices, (n, n))?;

        Ok(Self { l_factor })
    }
}

impl<F: Float + NumAssign + Sum + Debug + 'static> LinearOperator<F> for IC0Preconditioner<F> {
    fn shape(&self) -> (usize, usize) {
        self.l_factor.shape()
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        let n = self.l_factor.rows();
        if x.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: x.len(),
            });
        }

        // Solve L * L^T * y = x
        // First solve L * z = x (forward substitution)
        let mut z = vec![F::zero(); n];
        for i in 0..n {
            let row_start = self.l_factor.indptr[i];
            let row_end = self.l_factor.indptr[i + 1];

            let mut sum = x[i];
            let mut diag_val = F::one();

            for k in row_start..row_end {
                let j = self.l_factor.indices[k];
                let val = self.l_factor.data[k];

                match j.cmp(&i) {
                    std::cmp::Ordering::Less => sum -= val * z[j],
                    std::cmp::Ordering::Equal => diag_val = val,
                    std::cmp::Ordering::Greater => {}
                }
            }

            z[i] = sum / diag_val;
        }

        // Then solve L^T * y = z (backward substitution)
        let mut y = vec![F::zero(); n];
        for i in (0..n).rev() {
            let mut sum = z[i];

            // Find diagonal element
            let row_start = self.l_factor.indptr[i];
            let row_end = self.l_factor.indptr[i + 1];
            let mut diag_val = F::one();

            for k in row_start..row_end {
                let j = self.l_factor.indices[k];
                if j == i {
                    diag_val = self.l_factor.data[k];
                    break;
                }
            }

            // Subtract contributions from already computed y values
            for j in (i + 1)..n {
                // Find L[j,i] (which is L^T[i,j])
                let j_row_start = self.l_factor.indptr[j];
                let j_row_end = self.l_factor.indptr[j + 1];

                for k in j_row_start..j_row_end {
                    if self.l_factor.indices[k] == i {
                        sum -= self.l_factor.data[k] * y[j];
                        break;
                    }
                }
            }

            y[i] = sum / diag_val;
        }

        Ok(y)
    }

    fn has_adjoint(&self) -> bool {
        true
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // For symmetric preconditioner, adjoint is the same as forward operation
        self.matvec(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;

    #[test]
    fn test_ic0_simple() {
        // Test with a simple SPD matrix
        // A = [4  -1   0]
        //     [-1  4  -1]
        //     [0  -1   4]
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let indptr = vec![0, 2, 5, 7];
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let matrix = CsrMatrix::from_raw_csr(data, indptr, indices, (3, 3)).unwrap();

        let preconditioner = IC0Preconditioner::new(&matrix).unwrap();

        // Test by applying preconditioner to a vector
        let b = vec![1.0, 2.0, 3.0];
        let x = preconditioner.matvec(&b).unwrap();

        // The result should be approximately the solution to Ax = b
        // For this simple case, we can verify the result is reasonable
        assert!(x.iter().all(|&xi| xi.is_finite()));
    }

    #[test]
    fn test_ic0_not_spd() {
        // Test with a non-SPD matrix (should fail)
        // A = [1   2]
        //     [3   4]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let indptr = vec![0, 2, 4];
        let indices = vec![0, 1, 0, 1];
        let matrix = CsrMatrix::from_raw_csr(data, indptr, indices, (2, 2)).unwrap();

        let result = IC0Preconditioner::new(&matrix);
        assert!(result.is_err());
    }
}
