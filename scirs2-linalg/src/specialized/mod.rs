//! Specialized matrix implementations with optimized storage and operations
//!
//! This module provides implementations for special matrix types that can be stored
//! and operated on more efficiently than general matrices. These include:
//!
//! - Tridiagonal matrices: Matrices with non-zero elements only on the main diagonal
//!   and the diagonals immediately above and below it.
//! - Block tridiagonal matrices: Matrices composed of square blocks arranged in a
//!   tridiagonal pattern.
//! - Banded matrices: Matrices with non-zero elements only on diagonals within a
//!   certain distance from the main diagonal.
//! - Symmetric matrices: Matrices that are equal to their own transpose.
//!
//! These specialized matrices provide both memory and computational efficiency
//! for large matrices with special structure.

mod banded;
mod block_diagonal;
mod block_tridiagonal;
mod symmetric;
mod tridiagonal;

pub use self::banded::*;
pub use self::block_diagonal::*;
pub use self::block_tridiagonal::*;
pub use self::symmetric::*;
pub use self::tridiagonal::*;

// Export specific functions from block_tridiagonal for benchmark compatibility
pub use self::block_tridiagonal::block_tridiagonal_lu;

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use std::fmt::Debug;
use std::iter::Sum;

use crate::error::LinalgResult;
use crate::matrixfree::LinearOperator;

/// A trait for specialized matrices that enables efficient storage and operations
pub trait SpecializedMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// Return the number of rows in the matrix
    fn nrows(&self) -> usize;

    /// Return the number of columns in the matrix
    fn ncols(&self) -> usize;

    /// Get the element at position (i, j)
    fn get(&self, i: usize, j: usize) -> LinalgResult<A>;

    /// Perform matrix-vector multiplication: A * x
    fn matvec(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>>;

    /// Perform transposed matrix-vector multiplication: A^T * x
    fn matvec_transpose(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>>;

    /// Convert to a dense matrix representation
    fn to_dense(&self) -> LinalgResult<Array2<A>>;

    /// Convert to a matrix-free operator
    fn to_operator(&self) -> LinalgResult<LinearOperator<A>>
    where
        Self: Sync + 'static + Sized,
    {
        specialized_to_operator(self)
    }
}

/// Convert a specialized matrix to a matrix-free operator
#[allow(dead_code)]
pub fn specialized_to_operator<A, S>(matrix: &S) -> LinalgResult<LinearOperator<A>>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug + 'static,
    S: SpecializedMatrix<A> + Sync + 'static + Sized,
{
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();

    // Create a dense matrix clone to avoid lifetime issues
    let matrix_clone = matrix.to_dense()?;

    // Create a closure that captures the matrix clone and performs matrix-vector multiplication
    let matvec = move |x: &ArrayView1<A>| {
        let mut result = Array1::zeros(nrows);
        for i in 0..nrows {
            for j in 0..ncols {
                result[i] += matrix_clone[[i, j]] * x[j];
            }
        }
        result
    };

    // Create the linear operator
    if nrows == ncols {
        Ok(LinearOperator::new(nrows, matvec))
    } else {
        Ok(LinearOperator::new_rectangular(nrows, ncols, matvec))
    }
}
