//! Symmetric matrix implementation
//!
//! A symmetric matrix is a square matrix that is equal to its transpose.
//! This structure provides efficient storage and operations for such matrices.

use super::SpecializedMatrix;
use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::{Float, NumAssign, One, Zero};
use std::fmt::Debug;
use std::iter::Sum;

/// Symmetric matrix representation
///
/// A symmetric matrix has the property that A = A^T, meaning the element at position
/// (i, j) is equal to the element at position (j, i). This allows storing only
/// the lower or upper triangular part of the matrix, saving almost half the memory.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::specialized::SymmetricMatrix;
/// use scirs2_linalg::SpecializedMatrix;
///
/// // Create a 3x3 symmetric matrix
/// let a = array![
///     [1.0, 2.0, 3.0],
///     [2.0, 4.0, 5.0],
///     [3.0, 5.0, 6.0]
/// ];
///
/// let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();
///
/// // Get elements
/// assert_eq!(sym.get(0, 0).unwrap(), 1.0);
/// assert_eq!(sym.get(0, 1).unwrap(), 2.0);
/// assert_eq!(sym.get(1, 0).unwrap(), 2.0); // Equal to (0, 1)
/// assert_eq!(sym.get(2, 1).unwrap(), 5.0);
///
/// // Matrix-vector multiplication
/// let x = array![1.0, 2.0, 3.0];
/// let y = sym.matvec(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SymmetricMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// Storage for the lower triangular part of the matrix
    data: Array2<A>,
    /// Dimension of the matrix (n x n)
    n: usize,
}

impl<A> SymmetricMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// Create a new symmetric matrix from the lower triangular part
    ///
    /// # Arguments
    ///
    /// * `lower` - Lower triangular part of the matrix (including the diagonal)
    ///
    /// # Returns
    ///
    /// * `SymmetricMatrix` if the input is valid
    /// * `LinalgError` if the input is not a valid lower triangular part
    pub fn new(lower: ArrayView2<A>) -> LinalgResult<Self> {
        let n = lower.nrows();

        // Check that lower is at least a square matrix
        if lower.ncols() != n {
            return Err(LinalgError::ShapeError(format!(
                "Lower triangular part must be square, got shape {:?}",
                lower.shape()
            )));
        }

        // Verify that only the lower triangular part is filled
        // Not strictly necessary, but good for validation
        for i in 0..n {
            for j in i + 1..n {
                if lower[[i, j]] != A::zero() {
                    return Err(LinalgError::InvalidInputError(
                        "Lower triangular part must have zeros above the diagonal".to_string(),
                    ));
                }
            }
        }

        Ok(Self {
            data: lower.to_owned(),
            n,
        })
    }

    /// Create a new symmetric matrix from a general matrix
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix, expected to be symmetric
    ///
    /// # Returns
    ///
    /// * `SymmetricMatrix` if the input is symmetric
    /// * `LinalgError` if the input is not symmetric
    pub fn from_matrix(a: &ArrayView2<A>) -> LinalgResult<Self> {
        if a.nrows() != a.ncols() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix must be square to be symmetric, got shape {:?}",
                a.shape()
            )));
        }

        let n = a.nrows();

        // Check that the matrix is actually symmetric
        for i in 0..n {
            for j in i + 1..n {
                if (a[[i, j]] - a[[j, i]]).abs() > A::epsilon() {
                    return Err(LinalgError::InvalidInputError(format!(
                        "Matrix is not symmetric, a[{}, {}] != a[{}, {}]",
                        i, j, j, i
                    )));
                }
            }
        }

        // Extract the lower triangular part
        let mut lower = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                lower[[i, j]] = a[[i, j]];
            }
        }

        Ok(Self { data: lower, n })
    }

    /// Perform a Cholesky decomposition of the symmetric matrix
    ///
    /// The Cholesky decomposition of a symmetric positive-definite matrix A
    /// is a lower triangular matrix L such that A = L * L^T.
    ///
    /// # Returns
    ///
    /// * Lower triangular matrix L such that A = L * L^T
    /// * `LinalgError` if the matrix is not positive definite
    pub fn cholesky(&self) -> LinalgResult<Array2<A>> {
        let n = self.n;
        let mut l = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = A::zero();

                if j == i {
                    // Diagonal element
                    for k in 0..j {
                        sum += l[[j, k]] * l[[j, k]];
                    }

                    let diag_val = self.data[[i, j]] - sum;
                    if diag_val <= A::zero() {
                        return Err(LinalgError::InvalidInputError(
                            "Matrix is not positive definite".to_string(),
                        ));
                    }

                    l[[j, j]] = diag_val.sqrt();
                } else {
                    // Off-diagonal element
                    for k in 0..j {
                        sum += l[[i, k]] * l[[j, k]];
                    }

                    l[[i, j]] = (self.data[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        Ok(l)
    }

    /// Solve a symmetric system of equations Ax = b using Cholesky decomposition
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand side vector
    ///
    /// # Returns
    ///
    /// * Solution vector x
    /// * `LinalgError` if the matrix is not positive definite or dimensions are incompatible
    pub fn solve(&self, b: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if b.len() != self.n {
            return Err(LinalgError::ShapeError(format!(
                "Right-hand side length {} does not match matrix dimension {}",
                b.len(),
                self.n
            )));
        }

        // Perform Cholesky decomposition
        let l = self.cholesky()?;

        // Solve L * y = b (forward substitution)
        let mut y = Array1::zeros(self.n);
        for i in 0..self.n {
            let mut sum = A::zero();
            for j in 0..i {
                sum += l[[i, j]] * y[j];
            }
            y[i] = (b[i] - sum) / l[[i, i]];
        }

        // Solve L^T * x = y (backward substitution)
        let mut x = Array1::zeros(self.n);
        for i_rev in 0..self.n {
            let i = self.n - 1 - i_rev;
            let mut sum = A::zero();
            for j in i + 1..self.n {
                sum += l[[j, i]] * x[j];
            }
            x[i] = (y[i] - sum) / l[[i, i]];
        }

        Ok(x)
    }
}

impl<A> SpecializedMatrix<A> for SymmetricMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.n
    }

    fn get(&self, i: usize, j: usize) -> LinalgResult<A> {
        if i >= self.n || j >= self.n {
            return Err(LinalgError::IndexError(format!(
                "Index ({}, {}) out of bounds for matrix of size {}",
                i, j, self.n
            )));
        }

        // For symmetric matrix, we store only the lower triangular part
        // So we need to swap indices if j > i
        if j > i {
            Ok(self.data[[j, i]])
        } else {
            Ok(self.data[[i, j]])
        }
    }

    fn matvec(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if x.len() != self.n {
            return Err(LinalgError::ShapeError(format!(
                "Vector length {} does not match matrix dimension {}",
                x.len(),
                self.n
            )));
        }

        let mut y = Array1::zeros(self.n);

        // Optimized matrix-vector multiplication for symmetric matrices
        // Exploiting the symmetric structure to reduce the number of operations
        for i in 0..self.n {
            // Diagonal term
            y[i] += self.data[[i, i]] * x[i];

            // Off-diagonal terms in the lower triangular part
            for j in 0..i {
                let a_ij = self.data[[i, j]];
                y[i] += a_ij * x[j]; // A[i,j] * x[j]
                y[j] += a_ij * x[i]; // A[j,i] * x[i] = A[i,j] * x[i] (symmetry)
            }
        }

        Ok(y)
    }

    fn matvec_transpose(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        // For symmetric matrices, A^T = A, so matvec_transpose is the same as matvec
        self.matvec(x)
    }

    fn to_dense(&self) -> LinalgResult<Array2<A>> {
        let mut a = Array2::zeros((self.n, self.n));

        for i in 0..self.n {
            for j in 0..=i {
                let val = self.data[[i, j]];
                a[[i, j]] = val;

                // Fill the upper triangular part using symmetry
                if i != j {
                    a[[j, i]] = val;
                }
            }
        }

        Ok(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_symmetric_creation() {
        // Create a symmetric matrix
        let a = array![[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]];

        let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();

        assert_eq!(sym.nrows(), 3);
        assert_eq!(sym.ncols(), 3);

        // Check elements
        assert_relative_eq!(sym.get(0, 0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(sym.get(0, 1).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(sym.get(1, 0).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(sym.get(1, 1).unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(sym.get(1, 2).unwrap(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(sym.get(2, 1).unwrap(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(sym.get(2, 2).unwrap(), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_non_symmetric_error() {
        // Create a non-symmetric matrix
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let result = SymmetricMatrix::from_matrix(&a.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_matvec() {
        // Create a symmetric matrix
        let a = array![[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]];

        let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();

        let x = array![1.0, 2.0, 3.0];
        let y = sym.matvec(&x.view()).unwrap();

        // Expected: y = A * x
        let expected = array![
            1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0,
            2.0 * 1.0 + 4.0 * 2.0 + 5.0 * 3.0,
            3.0 * 1.0 + 5.0 * 2.0 + 6.0 * 3.0
        ];

        assert_eq!(y.len(), 3);
        assert_relative_eq!(y[0], expected[0], epsilon = 1e-10);
        assert_relative_eq!(y[1], expected[1], epsilon = 1e-10);
        assert_relative_eq!(y[2], expected[2], epsilon = 1e-10);
    }

    #[test]
    fn test_matvec_transpose() {
        // For symmetric matrices, matvec and matvec_transpose should give the same result
        let a = array![[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]];

        let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();

        let x = array![1.0, 2.0, 3.0];
        let y1 = sym.matvec(&x.view()).unwrap();
        let y2 = sym.matvec_transpose(&x.view()).unwrap();

        assert_eq!(y1.len(), 3);
        assert_eq!(y2.len(), 3);
        assert_relative_eq!(y1[0], y2[0], epsilon = 1e-10);
        assert_relative_eq!(y1[1], y2[1], epsilon = 1e-10);
        assert_relative_eq!(y1[2], y2[2], epsilon = 1e-10);
    }

    #[test]
    fn test_to_dense() {
        // Create a symmetric matrix
        let a = array![[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]];

        let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();

        let dense = sym.to_dense().unwrap();

        assert_eq!(dense.shape(), &[3, 3]);

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(dense[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_cholesky() {
        // Create a symmetric positive definite matrix
        let a = array![[4.0, 2.0, 1.0], [2.0, 3.0, 0.5], [1.0, 0.5, 6.0]];

        let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();

        let l = sym.cholesky().unwrap();

        // Verify the Cholesky decomposition by computing L * L^T
        let mut result = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    // If k > i or k > j, then L[i, k] or L[j, k] is zero
                    if k <= i.min(j) {
                        result[[i, j]] += l[[i, k]] * l[[j, k]];
                    }
                }
            }
        }

        // Check that L * L^T ≈ A
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(result[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_solve() {
        // Create a symmetric positive definite matrix
        let a = array![[4.0, 2.0, 1.0], [2.0, 3.0, 0.5], [1.0, 0.5, 6.0]];

        let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();

        // Right-hand side b = [1, 2, 3]
        let b = array![1.0, 2.0, 3.0];

        // Solve the system Ax = b
        let x = sym.solve(&b.view()).unwrap();

        // Verify the solution by calculating Ax
        let ax = sym.matvec(&x.view()).unwrap();

        // Check that Ax ≈ b
        assert_eq!(ax.len(), 3);
        assert_relative_eq!(ax[0], b[0], epsilon = 1e-10);
        assert_relative_eq!(ax[1], b[1], epsilon = 1e-10);
        assert_relative_eq!(ax[2], b[2], epsilon = 1e-10);
    }
}
