//! Eigenvalue solvers for banded matrices
//!
//! This module provides specialized solvers for banded matrices,
//! which are more efficient than general eigenvalue solvers.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::specialized::BandedMatrix;

/// Compute eigenvalues and eigenvectors of a symmetric banded matrix.
///
/// This algorithm transforms the banded matrix to tridiagonal form
/// and then applies the tridiagonal eigensolver.
///
/// # Arguments
///
/// * `a` - Input symmetric banded matrix
///
/// # Returns
///
/// * Tuple containing eigenvalues and eigenvectors
#[allow(dead_code)]
pub fn banded_eigh<F>(a: &ArrayView2<F>, bandwidth: usize) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float
        + NumAssign
        + Sum
        + One
        + Zero
        + 'static
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + std::fmt::Debug,
{
    let n = a.nrows();

    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    // Check if the matrix is symmetric
    // For banded matrices, we only need to check within the bandwidth
    for i in 0..n {
        for j in i + 1..std::cmp::min(i + bandwidth + 1, n) {
            let diff = (a[[i, j]] - a[[j, i]]).abs();
            if diff > F::epsilon() * F::from(10.0).unwrap() {
                return Err(LinalgError::ShapeError(
                    "Matrix must be symmetric for banded_eigh function".to_string(),
                ));
            }
        }
    }

    // Convert to BandedMatrix structure for efficient operations
    let _banded = BandedMatrix::frommatrix(a, bandwidth, bandwidth)?;

    // If bandwidth is 1, the matrix is already tridiagonal
    if bandwidth == 1 {
        let diag = Array1::from_iter((0..n).map(|i| a[[i, i]]));
        let subdiag = Array1::from_iter((0..n - 1).map(|i| a[[i + 1, i]]));

        // Use tridiagonal solver
        return crate::eigen_specialized::tridiagonal::tridiagonal_eigh(
            &diag.view(),
            &subdiag.view(),
        );
    }

    // For general banded matrices, reduce to tridiagonal form first
    // using a sequence of Givens rotations

    // Initialize diagonal and subdiagonal
    let mut diag = Array1::zeros(n);
    let mut subdiag = Array1::zeros(n - 1);

    // Initialize transformation matrix Q
    let mut q = Array2::eye(n);

    // Extract diagonal
    for i in 0..n {
        diag[i] = a[[i, i]];
    }

    // Create a copy of the matrix
    let mut h = a.to_owned();

    // Begin reduction to tridiagonal form
    for i in 0..n - 2 {
        for j in (i + 2)..std::cmp::min(i + bandwidth + 1, n) {
            if h[[j, i]].abs() > F::epsilon() {
                // Apply Givens rotation to eliminate h[j,i]
                let x = h[[j - 1, i]];
                let y = h[[j, i]];

                let r = (x * x + y * y).sqrt();
                let c = x / r;
                let s = -y / r;

                // Apply rotation to rows j-1 and j
                for k in i..std::cmp::min(j + bandwidth + 1, n) {
                    let temp = h[[j - 1, k]];
                    h[[j - 1, k]] = c * temp - s * h[[j, k]];
                    h[[j, k]] = s * temp + c * h[[j, k]];
                }

                // Apply rotation to columns j-1 and j
                for k in std::cmp::max(0, j - bandwidth - 1)..j + 1 {
                    let temp = h[[k, j - 1]];
                    h[[k, j - 1]] = c * temp - s * h[[k, j]];
                    h[[k, j]] = s * temp + c * h[[k, j]];
                }

                // Update transformation matrix Q
                for k in 0..n {
                    let temp = q[[k, j - 1]];
                    q[[k, j - 1]] = c * temp - s * q[[k, j]];
                    q[[k, j]] = s * temp + c * q[[k, j]];
                }
            }
        }
    }

    // Extract tridiagonal form
    for i in 0..n {
        diag[i] = h[[i, i]];
        if i < n - 1 {
            subdiag[i] = h[[i + 1, i]];
        }
    }

    // Solve tridiagonal eigenproblem
    let (eigenvalues, tri_vectors) =
        crate::eigen_specialized::tridiagonal::tridiagonal_eigh(&diag.view(), &subdiag.view())?;

    // Transform eigenvectors back
    let eigenvectors = q.dot(&tri_vectors);

    Ok((eigenvalues, eigenvectors))
}

/// Compute just the eigenvalues of a symmetric banded matrix.
///
/// This is similar to `banded_eigh` but only returns the eigenvalues.
///
/// # Arguments
///
/// * `a` - Input symmetric banded matrix
/// * `bandwidth` - Bandwidth of the matrix
///
/// # Returns
///
/// * Vector of eigenvalues
#[allow(dead_code)]
pub fn banded_eigvalsh<F>(a: &ArrayView2<F>, bandwidth: usize) -> LinalgResult<Array1<F>>
where
    F: Float
        + NumAssign
        + Sum
        + One
        + Zero
        + 'static
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + std::fmt::Debug,
{
    let n = a.nrows();

    // If bandwidth is 1, the matrix is already tridiagonal
    if bandwidth == 1 {
        let diag = Array1::from_iter((0..n).map(|i| a[[i, i]]));
        let subdiag = Array1::from_iter((0..n - 1).map(|i| a[[i + 1, i]]));

        // Use tridiagonal solver
        return crate::eigen_specialized::tridiagonal::tridiagonal_eigvalsh(
            &diag.view(),
            &subdiag.view(),
        );
    }

    // Otherwise, use the full solution and discard eigenvectors
    let (eigenvalues_) = banded_eigh(a, bandwidth)?;
    Ok(eigenvalues)
}
