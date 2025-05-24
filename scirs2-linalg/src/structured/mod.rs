//! Structured matrices support (Toeplitz, circulant) for efficient representations
//!
//! This module provides implementations of various structured matrices that have
//! special properties allowing for efficient storage and operations.
//!
//! # Overview
//!
//! Structured matrices are matrices with special patterns that allow them to be
//! represented using far fewer parameters than general matrices. This provides
//! significant memory savings and computational advantages for large matrices.
//!
//! ## Types of Structured Matrices
//!
//! * **Toeplitz matrices**: Matrices where each descending diagonal from left to right is constant
//! * **Circulant matrices**: Special Toeplitz matrices where each row is a cyclic shift of the first row
//! * **Hankel matrices**: Matrices where each ascending diagonal from left to right is constant
//!
//! # Examples
//!
//! ```
//! use ndarray::{Array1, array};
//! use scirs2_linalg::structured::{ToeplitzMatrix, StructuredMatrix};
//!
//! // Create a Toeplitz matrix from its first row and column
//! let first_row = array![1.0, 2.0, 3.0];
//! let first_col = array![1.0, 4.0, 5.0];
//!
//! let toeplitz = ToeplitzMatrix::new(first_row.view(), first_col.view()).unwrap();
//!
//! // The matrix represented is:
//! // [1.0, 2.0, 3.0]
//! // [4.0, 1.0, 2.0]
//! // [5.0, 4.0, 1.0]
//!
//! // Apply to a vector efficiently without forming the full matrix
//! let x = array![1.0, 2.0, 3.0];
//! let y = toeplitz.matvec(&x.view()).unwrap();
//! ```

use crate::error::LinalgResult;
use crate::matrixfree::LinearOperator;
use ndarray::ScalarOperand;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Float, NumAssign, One, Zero};
use std::{fmt::Debug, iter::Sum};

mod circulant;
mod hankel;
mod toeplitz;
mod utils;

pub use circulant::CirculantMatrix;
pub use hankel::HankelMatrix;
pub use toeplitz::ToeplitzMatrix;
pub use utils::{solve_circulant, solve_toeplitz};

/// A trait for structured matrices that can be represented efficiently
///
/// This trait defines common operations for all structured matrices.
pub trait StructuredMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// Returns the number of rows in the matrix
    fn nrows(&self) -> usize;

    /// Returns the number of columns in the matrix
    fn ncols(&self) -> usize;

    /// Returns the shape of the matrix as (rows, cols)
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Returns the element at position (i, j)
    fn get(&self, i: usize, j: usize) -> LinalgResult<A>;

    /// Multiply the matrix by a vector (matrix-vector product)
    fn matvec(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>>;

    /// Multiply the transpose of the matrix by a vector
    fn matvec_transpose(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>>;

    /// Convert the structured matrix to a dense ndarray representation
    fn to_dense(&self) -> LinalgResult<Array2<A>> {
        let (rows, cols) = self.shape();
        let mut result = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                result[[i, j]] = self.get(i, j)?;
            }
        }

        Ok(result)
    }

    /// Create a matrix-free operator from this structured matrix
    fn to_operator(&self) -> LinearOperator<A>
    where
        Self: Clone + Send + Sync + 'static,
    {
        let matrix = self.clone();
        let (rows, cols) = self.shape();

        LinearOperator::new_rectangular(rows, cols, move |x: &ArrayView1<A>| {
            matrix.matvec(x).unwrap()
        })
    }
}

/// Convert a structured matrix to a matrix-free operator
pub fn structured_to_operator<A, M>(matrix: &M) -> LinearOperator<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug + 'static,
    M: StructuredMatrix<A> + Clone + Send + Sync + 'static,
{
    let matrix_clone = matrix.clone();
    let (rows, cols) = matrix.shape();

    LinearOperator::new_rectangular(rows, cols, move |x: &ArrayView1<A>| {
        matrix_clone.matvec(x).unwrap()
    })
}

#[cfg(test)]
mod tests {
    // Tests are implemented in individual module test files
}
