//! Complex matrix operations
//!
//! This module provides functions for working with complex matrices.
//!
//! ## Basic Operations
//!
//! * Complex matrix multiplication
//! * Hermitian transpose (conjugate transpose)
//! * Complex matrix inverse
//! * Frobenius norm
//!
//! ## Enhanced Operations
//!
//! * Complex matrix determinant
//! * Matrix-vector multiplication
//! * Inner product for complex vectors
//! * Hermitian and unitary matrix validation
//! * Decompositions (Polar, Schur)
//! * Projections onto Hermitian and skew-Hermitian spaces
//! * Matrix exponential
//! * Matrix rank estimation
//! * Power method for eigenvalues

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::Complex;
use scirs2_core::numeric::{Float, One, Zero};
use std::fmt::Debug;

use scirs2_core::validation::check_square;

use crate::error::{LinalgError, LinalgResult};

pub mod enhanced_ops;
pub use enhanced_ops::*;

pub mod decompositions;
pub use decompositions::*;

/// Complex matrix multiplication C = A * B
#[allow(dead_code)]
pub fn complex_matmul<F>(
    a: &ArrayView2<Complex<F>>,
    b: &ArrayView2<Complex<F>>,
) -> LinalgResult<Array2<Complex<F>>>
where
    F: Float + Zero + One + Debug,
{
    if a.ncols() != b.nrows() {
        return Err(LinalgError::ShapeError(format!(
            "Incompatible dimensions for matrix multiplication: {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();

    let mut c = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex::zero();
            for l in 0..k {
                sum = sum + a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = sum;
        }
    }

    Ok(c)
}

/// Compute the Hermitian transpose (conjugate transpose) of a complex matrix
#[allow(dead_code)]
pub fn hermitian_transpose<F>(a: &ArrayView2<Complex<F>>) -> Array2<Complex<F>>
where
    F: Float,
{
    let (m, n) = (a.nrows(), a.ncols());
    let mut result = Array2::zeros((n, m));

    for i in 0..m {
        for j in 0..n {
            result[[j, i]] = a[[i, j]].conj();
        }
    }

    result
}

/// Frobenius norm of a complex matrix
#[allow(dead_code)]
pub fn complex_norm_frobenius<F>(a: &ArrayView2<Complex<F>>) -> F
where
    F: Float,
{
    let mut sum = F::zero();

    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let z = a[[i, j]];
            sum = sum + z.norm_sqr();
        }
    }

    sum.sqrt()
}

/// Compute the inverse of a complex matrix
#[allow(dead_code)]
pub fn complex_inverse<F>(a: &ArrayView2<Complex<F>>) -> LinalgResult<Array2<Complex<F>>>
where
    F: Float + Zero + One + Debug,
{
    // Check if the matrix is square using scirs2-core validation
    check_square(a, "matrix")?;

    let n = a.nrows();

    // Simple implementation for now - could be optimized
    // For larger matrices, we would use LU decomposition

    // Create augmented matrix [A|I]
    let mut augmented = Array2::zeros((n, 2 * n));

    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = a[[i, j]];
        }
        // Add identity matrix on the right
        augmented[[i, i + n]] = Complex::new(F::one(), F::zero());
    }

    // Gaussian elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = augmented[[i, i]].norm();

        for j in i + 1..n {
            let val = augmented[[j, i]].norm();
            if val > max_val {
                max_val = val;
                max_row = j;
            }
        }

        if max_val < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular and cannot be inverted".to_string(),
            ));
        }

        // Swap rows
        if max_row != i {
            for j in 0..2 * n {
                let temp = augmented[[i, j]];
                augmented[[i, j]] = augmented[[max_row, j]];
                augmented[[max_row, j]] = temp;
            }
        }

        // Scale pivot row
        let pivot = augmented[[i, i]];
        for j in 0..2 * n {
            augmented[[i, j]] = augmented[[i, j]] / pivot;
        }

        // Eliminate other rows
        for j in 0..n {
            if j != i {
                let factor = augmented[[j, i]];
                for k in 0..2 * n {
                    augmented[[j, k]] = augmented[[j, k]] - factor * augmented[[i, k]];
                }
            }
        }
    }

    // Extract inverse from right part of augmented matrix
    let mut inverse = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inverse[[i, j]] = augmented[[i, j + n]];
        }
    }

    Ok(inverse)
}
