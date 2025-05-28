//! Tridiagonal matrix implementation
//!
//! A tridiagonal matrix is a matrix where non-zero elements are only on the main
//! diagonal and the diagonals immediately above and below it. This structure
//! provides efficient storage and operations for such matrices.

use super::SpecializedMatrix;
use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::{Float, NumAssign, One, Zero};
use std::fmt::Debug;
use std::iter::Sum;

/// Tridiagonal matrix representation
///
/// A tridiagonal matrix stores only the three diagonals, giving O(n) storage
/// instead of O(n²) for dense matrices.
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array1};
/// use scirs2_linalg::specialized::TridiagonalMatrix;
/// use scirs2_linalg::SpecializedMatrix;
///
/// // Create a 4x4 tridiagonal matrix
/// let a = array![1.0, 2.0, 3.0, 4.0]; // Main diagonal
/// let b = array![5.0, 6.0, 7.0];      // Superdiagonal
/// let c = array![8.0, 9.0, 10.0];     // Subdiagonal
///
/// let tri = TridiagonalMatrix::new(a.view(), b.view(), c.view()).unwrap();
///
/// // The matrix is equivalent to:
/// // [[ 1.0, 5.0, 0.0, 0.0 ],
/// //  [ 8.0, 2.0, 6.0, 0.0 ],
/// //  [ 0.0, 9.0, 3.0, 7.0 ],
/// //  [ 0.0, 0.0, 10.0, 4.0 ]]
///
/// // Get elements
/// assert_eq!(tri.get(0, 0).unwrap(), 1.0);
/// assert_eq!(tri.get(0, 1).unwrap(), 5.0);
/// assert_eq!(tri.get(1, 0).unwrap(), 8.0);
/// assert_eq!(tri.get(2, 3).unwrap(), 7.0);
/// assert_eq!(tri.get(0, 2).unwrap(), 0.0); // Off-tridiagonal element
///
/// // Matrix-vector multiplication
/// let x = array![1.0, 2.0, 3.0, 4.0];
/// let y = tri.matvec(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TridiagonalMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// Main diagonal
    diag: Array1<A>,
    /// Superdiagonal (the diagonal above the main diagonal)
    superdiag: Array1<A>,
    /// Subdiagonal (the diagonal below the main diagonal)
    subdiag: Array1<A>,
    /// Number of rows and columns
    n: usize,
}

impl<A> TridiagonalMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// Create a new tridiagonal matrix from the three diagonals
    ///
    /// # Arguments
    ///
    /// * `diag` - Main diagonal of length n
    /// * `superdiag` - Superdiagonal of length n-1
    /// * `subdiag` - Subdiagonal of length n-1
    ///
    /// # Returns
    ///
    /// * `TridiagonalMatrix` if the diagonals have compatible lengths
    /// * `LinalgError` if the diagonal lengths are incompatible
    pub fn new(
        diag: ArrayView1<A>,
        superdiag: ArrayView1<A>,
        subdiag: ArrayView1<A>,
    ) -> LinalgResult<Self> {
        let n = diag.len();

        if superdiag.len() != n - 1 || subdiag.len() != n - 1 {
            return Err(LinalgError::ShapeError(format!(
                "Diagonal lengths are incompatible. Main diagonal: {}, superdiagonal: {}, subdiagonal: {}",
                n, superdiag.len(), subdiag.len()
            )));
        }

        Ok(Self {
            diag: diag.to_owned(),
            superdiag: superdiag.to_owned(),
            subdiag: subdiag.to_owned(),
            n,
        })
    }

    /// Create a new tridiagonal matrix from a general matrix, extracting the three diagonals
    ///
    /// # Arguments
    ///
    /// * `a` - Input square matrix to extract tridiagonal structure from
    ///
    /// # Returns
    ///
    /// * `TridiagonalMatrix` representation of the input matrix
    /// * `LinalgError` if the input matrix is not square
    pub fn from_matrix(a: &ArrayView2<A>) -> LinalgResult<Self> {
        if a.nrows() != a.ncols() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix must be square to convert to tridiagonal, got shape {:?}",
                a.shape()
            )));
        }

        let n = a.nrows();
        if n < 2 {
            // Special case for 1x1 matrix
            let mut diag = Array1::zeros(n);
            diag[0] = a[[0, 0]];

            return Ok(Self {
                diag,
                superdiag: Array1::zeros(0),
                subdiag: Array1::zeros(0),
                n,
            });
        }

        let mut diag = Array1::zeros(n);
        let mut superdiag = Array1::zeros(n - 1);
        let mut subdiag = Array1::zeros(n - 1);

        // Extract diagonals
        for i in 0..n {
            diag[i] = a[[i, i]];

            if i < n - 1 {
                superdiag[i] = a[[i, i + 1]];
                subdiag[i] = a[[i + 1, i]];
            }
        }

        Ok(Self {
            diag,
            superdiag,
            subdiag,
            n,
        })
    }

    /// Solve a tridiagonal system of equations Ax = b using the Thomas algorithm
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand side vector
    ///
    /// # Returns
    ///
    /// * Solution vector x
    /// * `LinalgError` if the matrix is singular or dimensions are incompatible
    pub fn solve(&self, b: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if b.len() != self.n {
            return Err(LinalgError::ShapeError(format!(
                "Right-hand side length {} does not match matrix dimension {}",
                b.len(),
                self.n
            )));
        }

        // Special case for 1x1 matrix
        if self.n == 1 {
            if self.diag[0].abs() < A::epsilon() {
                return Err(LinalgError::SingularMatrixError(
                    "Tridiagonal matrix is singular".to_string(),
                ));
            }
            let mut x = Array1::zeros(1);
            x[0] = b[0] / self.diag[0];
            return Ok(x);
        }

        // Implements the Thomas algorithm for tridiagonal systems
        let mut c_prime = Array1::zeros(self.n);
        let mut d_prime = Array1::zeros(self.n);

        // Forward sweep
        c_prime[0] = self.superdiag[0] / self.diag[0];
        d_prime[0] = b[0] / self.diag[0];

        for i in 1..self.n {
            let m = self.diag[i] - self.subdiag[i - 1] * c_prime[i - 1];

            if m.abs() < A::epsilon() {
                return Err(LinalgError::SingularMatrixError(
                    "Tridiagonal matrix is singular".to_string(),
                ));
            }

            if i < self.n - 1 {
                c_prime[i] = self.superdiag[i] / m;
            }

            d_prime[i] = (b[i] - self.subdiag[i - 1] * d_prime[i - 1]) / m;
        }

        // Back substitution
        let mut x = Array1::zeros(self.n);
        x[self.n - 1] = d_prime[self.n - 1];

        for i in (0..self.n - 1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }

        Ok(x)
    }
}

impl<A> SpecializedMatrix<A> for TridiagonalMatrix<A>
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

        // Main diagonal
        if i == j {
            return Ok(self.diag[i]);
        }

        // Superdiagonal
        if i + 1 == j {
            return Ok(self.superdiag[i]);
        }

        // Subdiagonal
        if i == j + 1 {
            return Ok(self.subdiag[j]);
        }

        // Zero elsewhere
        Ok(A::zero())
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

        // Special case for 1x1 matrix
        if self.n == 1 {
            y[0] = self.diag[0] * x[0];
            return Ok(y);
        }

        // First row
        y[0] = self.diag[0] * x[0] + self.superdiag[0] * x[1];

        // Middle rows
        for i in 1..self.n - 1 {
            y[i] =
                self.subdiag[i - 1] * x[i - 1] + self.diag[i] * x[i] + self.superdiag[i] * x[i + 1];
        }

        // Last row
        y[self.n - 1] =
            self.subdiag[self.n - 2] * x[self.n - 2] + self.diag[self.n - 1] * x[self.n - 1];

        Ok(y)
    }

    fn matvec_transpose(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if x.len() != self.n {
            return Err(LinalgError::ShapeError(format!(
                "Vector length {} does not match matrix dimension {}",
                x.len(),
                self.n
            )));
        }

        let mut y = Array1::zeros(self.n);

        // Special case for 1x1 matrix
        if self.n == 1 {
            y[0] = self.diag[0] * x[0];
            return Ok(y);
        }

        // First row (now using superdiag for the lower off-diagonal term)
        y[0] = self.diag[0] * x[0] + self.subdiag[0] * x[1];

        // Middle rows
        for i in 1..self.n - 1 {
            y[i] =
                self.superdiag[i - 1] * x[i - 1] + self.diag[i] * x[i] + self.subdiag[i] * x[i + 1];
        }

        // Last row
        y[self.n - 1] =
            self.superdiag[self.n - 2] * x[self.n - 2] + self.diag[self.n - 1] * x[self.n - 1];

        Ok(y)
    }

    fn to_dense(&self) -> LinalgResult<Array2<A>> {
        let mut a = Array2::zeros((self.n, self.n));

        // Set main diagonal
        for i in 0..self.n {
            a[[i, i]] = self.diag[i];
        }

        // Set superdiagonal
        for i in 0..self.n - 1 {
            a[[i, i + 1]] = self.superdiag[i];
        }

        // Set subdiagonal
        for i in 0..self.n - 1 {
            a[[i + 1, i]] = self.subdiag[i];
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
    fn test_tridiagonal_creation() {
        let diag = array![1.0, 2.0, 3.0, 4.0];
        let superdiag = array![5.0, 6.0, 7.0];
        let subdiag = array![8.0, 9.0, 10.0];

        let tri = TridiagonalMatrix::new(diag.view(), superdiag.view(), subdiag.view()).unwrap();

        assert_eq!(tri.nrows(), 4);
        assert_eq!(tri.ncols(), 4);

        // Check elements
        assert_relative_eq!(tri.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(tri.get(0, 1).unwrap(), 5.0);
        assert_relative_eq!(tri.get(1, 0).unwrap(), 8.0);
        assert_relative_eq!(tri.get(1, 1).unwrap(), 2.0);
        assert_relative_eq!(tri.get(1, 2).unwrap(), 6.0);
        assert_relative_eq!(tri.get(2, 1).unwrap(), 9.0);

        // Check zero elements
        assert_relative_eq!(tri.get(0, 2).unwrap(), 0.0);
        assert_relative_eq!(tri.get(0, 3).unwrap(), 0.0);
        assert_relative_eq!(tri.get(2, 0).unwrap(), 0.0);
    }

    #[test]
    fn test_from_matrix() {
        let a = array![
            [1.0, 5.0, 0.0, 0.0],
            [8.0, 2.0, 6.0, 0.0],
            [0.0, 9.0, 3.0, 7.0],
            [0.0, 0.0, 10.0, 4.0]
        ];

        let tri = TridiagonalMatrix::from_matrix(&a.view()).unwrap();

        assert_eq!(tri.nrows(), 4);
        assert_eq!(tri.ncols(), 4);

        // Check diagonals
        assert_relative_eq!(tri.diag[0], 1.0);
        assert_relative_eq!(tri.diag[1], 2.0);
        assert_relative_eq!(tri.diag[2], 3.0);
        assert_relative_eq!(tri.diag[3], 4.0);

        assert_relative_eq!(tri.superdiag[0], 5.0);
        assert_relative_eq!(tri.superdiag[1], 6.0);
        assert_relative_eq!(tri.superdiag[2], 7.0);

        assert_relative_eq!(tri.subdiag[0], 8.0);
        assert_relative_eq!(tri.subdiag[1], 9.0);
        assert_relative_eq!(tri.subdiag[2], 10.0);
    }

    #[test]
    fn test_matvec() {
        let diag = array![1.0, 2.0, 3.0, 4.0];
        let superdiag = array![5.0, 6.0, 7.0];
        let subdiag = array![8.0, 9.0, 10.0];

        let tri = TridiagonalMatrix::new(diag.view(), superdiag.view(), subdiag.view()).unwrap();

        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = tri.matvec(&x.view()).unwrap();

        // Expected result: y = A * x
        let expected = array![
            1.0 * 1.0 + 5.0 * 2.0,
            8.0 * 1.0 + 2.0 * 2.0 + 6.0 * 3.0,
            9.0 * 2.0 + 3.0 * 3.0 + 7.0 * 4.0,
            10.0 * 3.0 + 4.0 * 4.0
        ];

        assert_eq!(y.len(), 4);
        assert_relative_eq!(y[0], expected[0], epsilon = 1e-10);
        assert_relative_eq!(y[1], expected[1], epsilon = 1e-10);
        assert_relative_eq!(y[2], expected[2], epsilon = 1e-10);
        assert_relative_eq!(y[3], expected[3], epsilon = 1e-10);
    }

    #[test]
    fn test_matvec_transpose() {
        let diag = array![1.0, 2.0, 3.0, 4.0];
        let superdiag = array![5.0, 6.0, 7.0];
        let subdiag = array![8.0, 9.0, 10.0];

        let tri = TridiagonalMatrix::new(diag.view(), superdiag.view(), subdiag.view()).unwrap();

        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = tri.matvec_transpose(&x.view()).unwrap();

        // Expected result: y = A^T * x
        let expected = array![
            1.0 * 1.0 + 8.0 * 2.0,
            5.0 * 1.0 + 2.0 * 2.0 + 9.0 * 3.0,
            6.0 * 2.0 + 3.0 * 3.0 + 10.0 * 4.0,
            7.0 * 3.0 + 4.0 * 4.0
        ];

        assert_eq!(y.len(), 4);
        assert_relative_eq!(y[0], expected[0], epsilon = 1e-10);
        assert_relative_eq!(y[1], expected[1], epsilon = 1e-10);
        assert_relative_eq!(y[2], expected[2], epsilon = 1e-10);
        assert_relative_eq!(y[3], expected[3], epsilon = 1e-10);
    }

    #[test]
    fn test_to_dense() {
        let diag = array![1.0, 2.0, 3.0];
        let superdiag = array![4.0, 5.0];
        let subdiag = array![6.0, 7.0];

        let tri = TridiagonalMatrix::new(diag.view(), superdiag.view(), subdiag.view()).unwrap();

        let dense = tri.to_dense().unwrap();

        let expected = array![[1.0, 4.0, 0.0], [6.0, 2.0, 5.0], [0.0, 7.0, 3.0]];

        assert_eq!(dense.shape(), &[3, 3]);

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(dense[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_solve() {
        // Create a simple tridiagonal system
        let diag = array![2.0, 2.0, 2.0, 2.0];
        let superdiag = array![-1.0, -1.0, -1.0];
        let subdiag = array![-1.0, -1.0, -1.0];

        let tri = TridiagonalMatrix::new(diag.view(), superdiag.view(), subdiag.view()).unwrap();

        // Right-hand side b = [1, 2, 3, 4]
        let b = array![1.0, 2.0, 3.0, 4.0];

        // Solve the system Ax = b
        let x = tri.solve(&b.view()).unwrap();

        // Verify the solution by calculating Ax
        let ax = tri.matvec(&x.view()).unwrap();

        // Check that Ax ≈ b
        assert_eq!(ax.len(), 4);
        assert_relative_eq!(ax[0], b[0], epsilon = 1e-10);
        assert_relative_eq!(ax[1], b[1], epsilon = 1e-10);
        assert_relative_eq!(ax[2], b[2], epsilon = 1e-10);
        assert_relative_eq!(ax[3], b[3], epsilon = 1e-10);
    }
}
