//! Preconditioners for iterative solvers

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use crate::linalg::interface::LinearOperator;
use num_traits::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

/// Jacobi (diagonal) preconditioner
///
/// This preconditioner uses the inverse of the diagonal of the matrix.
/// M^(-1) = diag(1/a_11, 1/a_22, ..., 1/a_nn)
pub struct JacobiPreconditioner<F> {
    inv_diagonal: Vec<F>,
    size: usize,
}

impl<F: Float> JacobiPreconditioner<F> {
    /// Create a new Jacobi preconditioner from a sparse matrix
    pub fn new(matrix: &CsrMatrix<F>) -> SparseResult<Self>
    where
        F: Debug,
    {
        let n = matrix.rows();
        let mut inv_diagonal = vec![F::zero(); n];

        // Extract diagonal elements
        for i in 0..n {
            let diag_val = matrix.get(i, i);
            if diag_val.abs() < F::epsilon() {
                return Err(SparseError::ValueError(format!(
                    "Zero diagonal element at position {}",
                    i
                )));
            }
            inv_diagonal[i] = F::one() / diag_val;
        }

        Ok(Self {
            inv_diagonal,
            size: n,
        })
    }

    /// Create from diagonal values directly
    pub fn from_diagonal(diagonal: Vec<F>) -> SparseResult<Self> {
        let size = diagonal.len();
        let mut inv_diagonal = vec![F::zero(); size];

        for (i, &d) in diagonal.iter().enumerate() {
            if d.abs() < F::epsilon() {
                return Err(SparseError::ValueError(format!(
                    "Zero diagonal element at position {}",
                    i
                )));
            }
            inv_diagonal[i] = F::one() / d;
        }

        Ok(Self { inv_diagonal, size })
    }
}

impl<F: Float + NumAssign> LinearOperator<F> for JacobiPreconditioner<F> {
    fn shape(&self) -> (usize, usize) {
        (self.size, self.size)
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.size {
            return Err(SparseError::DimensionMismatch {
                expected: self.size,
                found: x.len(),
            });
        }

        Ok(x.iter()
            .zip(&self.inv_diagonal)
            .map(|(&xi, &di)| xi * di)
            .collect())
    }

    fn has_adjoint(&self) -> bool {
        true
    }

    fn rmatvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        // For real diagonal matrices, adjoint is the same as forward operation
        self.matvec(x)
    }
}

/// Symmetric Successive Over-Relaxation (SSOR) preconditioner
///
/// This preconditioner uses the SSOR method with a relaxation parameter omega.
/// M = (D + ωL) * D^(-1) * (D + ωU)
pub struct SSORPreconditioner<F> {
    matrix: CsrMatrix<F>,
    omega: F,
    diagonal: Vec<F>,
}

impl<F: Float + NumAssign + Sum + Debug + 'static> SSORPreconditioner<F> {
    /// Create a new SSOR preconditioner
    pub fn new(matrix: CsrMatrix<F>, omega: F) -> SparseResult<Self> {
        if omega <= F::zero() || omega >= F::from(2.0).unwrap() {
            return Err(SparseError::ValueError(
                "Relaxation parameter omega must be in (0, 2)".to_string(),
            ));
        }

        let n = matrix.rows();
        let mut diagonal = vec![F::zero(); n];

        // Extract diagonal elements
        for i in 0..n {
            diagonal[i] = matrix.get(i, i);
            if diagonal[i].abs() < F::epsilon() {
                return Err(SparseError::ValueError(format!(
                    "Zero diagonal element at position {}",
                    i
                )));
            }
        }

        Ok(Self {
            matrix,
            omega,
            diagonal,
        })
    }
}

impl<F: Float + NumAssign + Sum + Debug + 'static> LinearOperator<F> for SSORPreconditioner<F> {
    fn shape(&self) -> (usize, usize) {
        self.matrix.shape()
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        let n = self.matrix.rows();
        if x.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: x.len(),
            });
        }

        // Forward sweep: solve (D + ωL)y = x
        let mut y = vec![F::zero(); n];
        for i in 0..n {
            let mut sum = x[i];
            let row_range = self.matrix.row_range(i);
            let row_indices = &self.matrix.indices[row_range.clone()];
            let row_data = &self.matrix.data[row_range];

            for (idx, &j) in row_indices.iter().enumerate() {
                if j < i {
                    sum -= self.omega * row_data[idx] * y[j];
                }
            }
            y[i] = sum / self.diagonal[i];
        }

        // Diagonal scaling: z = D^(-1) * y
        let mut z = vec![F::zero(); n];
        for i in 0..n {
            z[i] = y[i] * self.diagonal[i];
        }

        // Backward sweep: solve (D + ωU)w = z
        let mut w = vec![F::zero(); n];
        for i in (0..n).rev() {
            let mut sum = z[i];
            let row_range = self.matrix.row_range(i);
            let row_indices = &self.matrix.indices[row_range.clone()];
            let row_data = &self.matrix.data[row_range];

            for (idx, &j) in row_indices.iter().enumerate() {
                if j > i {
                    sum -= self.omega * row_data[idx] * w[j];
                }
            }
            w[i] = sum / self.diagonal[i];
        }

        Ok(w)
    }

    fn has_adjoint(&self) -> bool {
        false // SSOR is not generally self-adjoint
    }
}

/// Incomplete LU factorization with zero fill-in (ILU(0))
///
/// This preconditioner computes an incomplete LU factorization
/// where L and U have the same sparsity pattern as the original matrix.
pub struct ILU0Preconditioner<F> {
    l_data: Vec<F>,
    u_data: Vec<F>,
    l_indices: Vec<usize>,
    u_indices: Vec<usize>,
    l_indptr: Vec<usize>,
    u_indptr: Vec<usize>,
    n: usize,
}

impl<F: Float + NumAssign + Sum + Debug + 'static> ILU0Preconditioner<F> {
    /// Create a new ILU(0) preconditioner
    pub fn new(matrix: &CsrMatrix<F>) -> SparseResult<Self> {
        let n = matrix.rows();

        // Copy the matrix data for modification
        let mut data = matrix.data.clone();
        let indices = matrix.indices.clone();
        let indptr = matrix.indptr.clone();

        // Perform ILU(0) factorization
        for k in 0..n {
            let k_diag_idx = find_diagonal_index(&indices, &indptr, k)?;
            let k_diag = data[k_diag_idx];

            if k_diag.abs() < F::epsilon() {
                return Err(SparseError::ValueError(format!(
                    "Zero pivot at position {}",
                    k
                )));
            }

            // Update rows k+1 to n-1
            for i in (k + 1)..n {
                let row_start = indptr[i];
                let row_end = indptr[i + 1];

                // Find position of k in row i
                let mut k_pos = None;
                for j in row_start..row_end {
                    if indices[j] == k {
                        k_pos = Some(j);
                        break;
                    }
                    if indices[j] > k {
                        break;
                    }
                }

                if let Some(ki_idx) = k_pos {
                    // Compute multiplier
                    let mult = data[ki_idx] / k_diag;
                    data[ki_idx] = mult;

                    // Update the rest of row i
                    let k_row_start = indptr[k];
                    let k_row_end = indptr[k + 1];

                    for kj_idx in k_row_start..k_row_end {
                        let j = indices[kj_idx];
                        if j <= k {
                            continue;
                        }

                        // Find position of j in row i
                        for ij_idx in row_start..row_end {
                            if indices[ij_idx] == j {
                                data[ij_idx] = data[ij_idx] - mult * data[kj_idx];
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Split into L and U parts
        let mut l_data = Vec::new();
        let mut u_data = Vec::new();
        let mut l_indices = Vec::new();
        let mut u_indices = Vec::new();
        let mut l_indptr = vec![0];
        let mut u_indptr = vec![0];

        for i in 0..n {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];

            for j in row_start..row_end {
                let col = indices[j];
                let val = data[j];

                match col.cmp(&i) {
                    std::cmp::Ordering::Less => {
                        // Lower triangular part
                        l_indices.push(col);
                        l_data.push(val);
                    }
                    std::cmp::Ordering::Equal => {
                        // Diagonal (goes to U)
                        u_indices.push(col);
                        u_data.push(val);
                    }
                    std::cmp::Ordering::Greater => {
                        // Upper triangular part
                        u_indices.push(col);
                        u_data.push(val);
                    }
                }
            }

            l_indptr.push(l_indices.len());
            u_indptr.push(u_indices.len());
        }

        Ok(Self {
            l_data,
            u_data,
            l_indices,
            u_indices,
            l_indptr,
            u_indptr,
            n,
        })
    }
}

impl<F: Float + NumAssign + Sum + Debug + 'static> LinearOperator<F> for ILU0Preconditioner<F> {
    fn shape(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn matvec(&self, x: &[F]) -> SparseResult<Vec<F>> {
        if x.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: x.len(),
            });
        }

        // Solve Ly = x (forward substitution)
        let mut y = vec![F::zero(); self.n];
        for i in 0..self.n {
            y[i] = x[i];
            let row_start = self.l_indptr[i];
            let row_end = self.l_indptr[i + 1];

            for j in row_start..row_end {
                let col = self.l_indices[j];
                y[i] = y[i] - self.l_data[j] * y[col];
            }
        }

        // Solve Uz = y (backward substitution)
        let mut z = vec![F::zero(); self.n];
        for i in (0..self.n).rev() {
            z[i] = y[i];
            let row_start = self.u_indptr[i];
            let row_end = self.u_indptr[i + 1];

            let mut diag_val = F::one();
            for j in row_start..row_end {
                let col = self.u_indices[j];
                match col.cmp(&i) {
                    std::cmp::Ordering::Equal => diag_val = self.u_data[j],
                    std::cmp::Ordering::Greater => z[i] = z[i] - self.u_data[j] * z[col],
                    std::cmp::Ordering::Less => {}
                }
            }

            z[i] /= diag_val;
        }

        Ok(z)
    }

    fn has_adjoint(&self) -> bool {
        false // ILU is not generally self-adjoint
    }
}

// Helper function to find diagonal index in CSR format
fn find_diagonal_index(indices: &[usize], indptr: &[usize], row: usize) -> SparseResult<usize> {
    let row_start = indptr[row];
    let row_end = indptr[row + 1];

    for idx in row_start..row_end {
        if indices[idx] == row {
            return Ok(idx);
        }
    }

    Err(SparseError::ValueError(format!(
        "Missing diagonal element at row {}",
        row
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobi_preconditioner() {
        // Create a diagonal matrix
        let diagonal = vec![2.0, 3.0, 4.0];
        let precond = JacobiPreconditioner::from_diagonal(diagonal).unwrap();

        // Test application
        let x = vec![2.0, 6.0, 12.0];
        let result = precond.matvec(&x).unwrap();

        // Should be [1.0, 2.0, 3.0]
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
    }
}
