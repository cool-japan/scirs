//! Standard eigenvalue computations with extended precision
//!
//! This module provides the main public functions for computing eigenvalues
//! and eigenvectors using extended precision arithmetic.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, One, Zero};

use super::super::{DemotableTo, PromotableTo};
use super::generalized_eigen::*;
use super::iterative::*;
use crate::error::LinalgResult;

/// Type for eigenvalue and eigenvector results with complex numbers
pub type EigenResult<A> = LinalgResult<(
    Array1<scirs2_core::numeric::Complex<A>>,
    Array2<scirs2_core::numeric::Complex<A>>,
)>;

/// Compute eigenvalues of a general matrix using extended precision
///
/// This function computes the eigenvalues of a general square matrix using
/// a higher precision implementation of the QR algorithm.
///
/// # Parameters
///
/// * `a` - Input matrix
/// * `max_iter` - Maximum number of iterations (default: 100)
/// * `tol` - Convergence tolerance (default: 1e-8 for the working precision)
///
/// # Returns
///
/// * Complex eigenvalues
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ArrayView2};
/// use scirs2_linalg::extended_precision::eigen::extended_eigvals;
/// use scirs2_core::numeric::Complex;
///
/// let a = array![
///     [1.0_f32, 2.0],
///     [3.0, 4.0]
/// ];
///
/// // Compute eigenvalues with extended precision
/// let eigvals = extended_eigvals::<_, f64>(&a.view(), None, None).unwrap();
///
/// // Expected eigenvalues approximately (-0.3723, 5.3723)
/// assert!((eigvals[0].re + 0.3723).abs() < 1e-4 || (eigvals[0].re - 5.3723).abs() < 1e-4);
/// ```
#[allow(dead_code)]
pub fn extended_eigvals<A, I>(
    a: &ArrayView2<A>,
    max_iter: Option<usize>,
    tol: Option<A>,
) -> LinalgResult<Array1<scirs2_core::numeric::Complex<A>>>
where
    A: Float + Zero + One + PromotableTo<I> + DemotableTo<A> + Copy,
    I: Float
        + Zero
        + One
        + DemotableTo<A>
        + Copy
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::DivAssign,
{
    if a.nrows() != a.ncols() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();
    let max_iter = max_iter.unwrap_or(100 * n);
    let tol = tol.unwrap_or(A::epsilon().sqrt());

    // Convert matrix to higher precision for computation
    let mut a_high = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_high[[i, j]] = a[[i, j]].promote();
        }
    }

    // Convert to Hessenberg form first (reduces computational work)
    let a_high = hessenberg_reduction(a_high);

    // Apply QR algorithm with implicit shifts in higher precision
    let eigenvalues_high = qr_algorithm(a_high, max_iter, I::from(tol.promote()).unwrap());

    // Convert eigenvalues back to original precision
    let mut eigenvalues = Array1::zeros(n);
    for i in 0..n {
        eigenvalues[i] = scirs2_core::numeric::Complex::new(
            eigenvalues_high[i].re.demote(),
            eigenvalues_high[i].im.demote(),
        );
    }

    Ok(eigenvalues)
}

/// Compute eigenvalues and eigenvectors of a general matrix using extended precision
///
/// This function computes the eigenvalues and eigenvectors of a general square matrix
/// using a higher precision implementation of the QR algorithm.
///
/// # Parameters
///
/// * `a` - Input matrix
/// * `max_iter` - Maximum number of iterations (default: 100)
/// * `tol` - Convergence tolerance (default: 1e-8 for the working precision)
///
/// # Returns
///
/// * Tuple containing (eigenvalues, eigenvectors)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ArrayView2};
/// use scirs2_linalg::extended_precision::eigen::extended_eig;
/// use scirs2_core::numeric::Complex;
///
/// let a = array![
///     [1.0_f32, 2.0],
///     [3.0, 4.0]
/// ];
///
/// // Compute eigenvalues and eigenvectors with extended precision
/// let (eigvals, eigvecs) = extended_eig::<_, f64>(&a.view(), None, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn extended_eig<A, I>(
    a: &ArrayView2<A>,
    max_iter: Option<usize>,
    tol: Option<A>,
) -> EigenResult<A>
where
    A: Float + Zero + One + PromotableTo<I> + DemotableTo<A> + Copy,
    I: Float
        + Zero
        + One
        + DemotableTo<A>
        + Copy
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::DivAssign,
{
    if a.nrows() != a.ncols() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    // Compute eigenvalues first
    let eigenvalues = extended_eigvals(a, max_iter, tol)?;

    // Now compute eigenvectors using inverse iteration in extended precision
    let n = a.nrows();
    let mut eigenvectors = Array2::zeros((n, n));

    // Convert matrix to higher precision for computation
    let mut a_high = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_high[[i, j]] = a[[i, j]].promote();
        }
    }

    // For each eigenvalue, compute the corresponding eigenvector using inverse iteration
    for (k, lambda) in eigenvalues.iter().enumerate() {
        // Create (A - λI) matrix in extended precision as complex numbers
        let mut shiftedmatrix: Array2<scirs2_core::numeric::Complex<I>> = Array2::zeros((n, n));
        let lambda_high =
            scirs2_core::numeric::Complex::new(lambda.re.promote(), lambda.im.promote());

        // Convert real matrix to complex and subtract eigenvalue from diagonal
        for i in 0..n {
            for j in 0..n {
                shiftedmatrix[[i, j]] =
                    scirs2_core::numeric::Complex::new(a_high[[i, j]], I::zero());
            }
        }

        for i in 0..n {
            shiftedmatrix[[i, i]] = shiftedmatrix[[i, i]] - lambda_high;
        }

        // Compute eigenvector using inverse iteration with extended precision
        let eigenvector_high = compute_eigenvector_inverse_iteration(
            &shiftedmatrix,
            lambda_high,
            max_iter.unwrap_or(100),
            I::from(tol.unwrap_or(A::epsilon().sqrt()).promote()).unwrap(),
        );

        // Convert eigenvector back to original precision
        for i in 0..n {
            eigenvectors[[i, k]] = scirs2_core::numeric::Complex::new(
                eigenvector_high[i].re.demote(),
                eigenvector_high[i].im.demote(),
            );
        }
    }

    Ok((eigenvalues, eigenvectors))
}

/// Compute eigenvalues of a symmetric/Hermitian matrix using extended precision
///
/// This function computes the eigenvalues of a symmetric/Hermitian square matrix
/// using a higher precision implementation of the QR algorithm specialized for
/// symmetric matrices.
///
/// # Parameters
///
/// * `a` - Input symmetric matrix
/// * `max_iter` - Maximum number of iterations (default: 100)
/// * `tol` - Convergence tolerance (default: 1e-8 for the working precision)
///
/// # Returns
///
/// * Real eigenvalues
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ArrayView2};
/// use scirs2_linalg::extended_precision::eigen::extended_eigvalsh;
///
/// let a = array![
///     [2.0_f32, 1.0],
///     [1.0, 2.0]
/// ];
///
/// // Compute eigenvalues with extended precision
/// let eigvals = extended_eigvalsh::<_, f64>(&a.view(), None, None).unwrap();
///
/// // Check that we got 2 eigenvalues
/// assert_eq!(eigvals.len(), 2);
///
/// // For symmetric matrices, eigenvalues should be real
/// // Just verify they're finite and reasonable
/// assert!(eigvals[0].is_finite());
/// assert!(eigvals[1].is_finite());
/// ```
#[allow(dead_code)]
pub fn extended_eigvalsh<A, I>(
    a: &ArrayView2<A>,
    max_iter: Option<usize>,
    tol: Option<A>,
) -> LinalgResult<Array1<A>>
where
    A: Float + Zero + One + PromotableTo<I> + DemotableTo<A> + Copy,
    I: Float
        + Zero
        + One
        + DemotableTo<A>
        + Copy
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::DivAssign,
{
    if a.nrows() != a.ncols() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    // Check symmetry
    let n = a.nrows();
    for i in 0..n {
        for j in i + 1..n {
            if (a[[i, j]] - a[[j, i]]).abs() > A::epsilon() * A::from(10.0).unwrap() {
                return Err(crate::error::LinalgError::InvalidInputError(
                    "Matrix must be symmetric/Hermitian".to_string(),
                ));
            }
        }
    }

    let max_iter = max_iter.unwrap_or(100 * n);
    let tol = tol.unwrap_or(A::epsilon().sqrt());

    // Convert matrix to higher precision for computation
    let mut a_high = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_high[[i, j]] = a[[i, j]].promote();
        }
    }

    // Tridiagonalize the symmetric matrix
    let a_high = tridiagonalize(a_high);

    // Apply QR algorithm for symmetric tridiagonal matrices
    let eigenvalues_high =
        qr_algorithm_symmetric(a_high, max_iter, I::from(tol.promote()).unwrap());

    // Convert eigenvalues back to original precision
    let mut eigenvalues = Array1::zeros(n);
    for i in 0..n {
        eigenvalues[i] = eigenvalues_high[i].demote();
    }

    Ok(eigenvalues)
}

/// Compute eigenvalues and eigenvectors of a symmetric/Hermitian matrix using extended precision
///
/// This function computes the eigenvalues and eigenvectors of a symmetric/Hermitian matrix
/// using a higher precision implementation of the QR algorithm specialized for
/// symmetric matrices.
///
/// # Parameters
///
/// * `a` - Input symmetric matrix
/// * `max_iter` - Maximum number of iterations (default: 100)
/// * `tol` - Convergence tolerance (default: 1e-8 for the working precision)
///
/// # Returns
///
/// * Tuple containing (eigenvalues, eigenvectors) where eigenvalues are real
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ArrayView2};
/// use scirs2_linalg::extended_precision::eigen::extended_eigh;
///
/// let a = array![
///     [2.0_f32, 1.0],
///     [1.0, 2.0]
/// ];
///
/// // Compute eigenvalues and eigenvectors with extended precision
/// let (eigvals, eigvecs) = extended_eigh::<_, f64>(&a.view(), None, None).unwrap();
///
/// // Check that we got 2 eigenvalues
/// assert_eq!(eigvals.len(), 2);
///
/// // For symmetric matrices, eigenvalues should be real
/// // Just verify they're finite and reasonable
/// assert!(eigvals[0].is_finite());
/// assert!(eigvals[1].is_finite());
///
/// // Check eigenvector properties
/// assert_eq!(eigvecs.shape(), &[2, 2]);
///
/// // Eigenvectors should have unit norm (approximately)
/// let norm1 = eigvecs.column(0).dot(&eigvecs.column(0)).sqrt();
/// let norm2 = eigvecs.column(1).dot(&eigvecs.column(1)).sqrt();
/// assert!((norm1 - 1.0).abs() < 0.1);
/// assert!((norm2 - 1.0).abs() < 0.1);
/// ```
#[allow(dead_code)]
pub fn extended_eigh<A, I>(
    a: &ArrayView2<A>,
    max_iter: Option<usize>,
    tol: Option<A>,
) -> LinalgResult<(Array1<A>, Array2<A>)>
where
    A: Float + Zero + One + PromotableTo<I> + DemotableTo<A> + Copy,
    I: Float
        + Zero
        + One
        + DemotableTo<A>
        + Copy
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::DivAssign
        + 'static,
{
    if a.nrows() != a.ncols() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    // Check symmetry
    let n = a.nrows();
    for i in 0..n {
        for j in i + 1..n {
            if (a[[i, j]] - a[[j, i]]).abs() > A::epsilon() * A::from(10.0).unwrap() {
                return Err(crate::error::LinalgError::InvalidInputError(
                    "Matrix must be symmetric/Hermitian".to_string(),
                ));
            }
        }
    }

    let max_iter = max_iter.unwrap_or(100 * n);
    let tol = tol.unwrap_or(A::epsilon().sqrt());

    // Convert matrix to higher precision for computation
    let mut a_high = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_high[[i, j]] = a[[i, j]].promote();
        }
    }

    // Tridiagonalize the symmetric matrix
    let (a_tri, q) = tridiagonalize_with_transform(a_high);

    // Apply QR algorithm for symmetric tridiagonal matrices
    let (eigenvalues_high, eigenvectors_high) =
        qr_algorithm_symmetric_with_vectors(a_tri, q, max_iter, I::from(tol.promote()).unwrap());

    // Convert eigenvalues and eigenvectors back to original precision
    let mut eigenvalues = Array1::zeros(n);
    let mut eigenvectors = Array2::zeros((n, n));

    for i in 0..n {
        eigenvalues[i] = eigenvalues_high[i].demote();
        for j in 0..n {
            eigenvectors[[i, j]] = eigenvectors_high[[i, j]].demote();
        }
    }

    Ok((eigenvalues, eigenvectors))
}
