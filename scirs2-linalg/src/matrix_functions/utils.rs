//! Utility functions shared across matrix function modules

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, One};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

/// Check if a floating point number is close to an integer
pub fn is_integer<F: Float>(x: F) -> bool {
    (x - x.round()).abs() < F::from(1e-10).unwrap_or(F::epsilon())
}

/// Check if a matrix is diagonal
pub fn is_diagonal<F>(a: &ArrayView2<F>) -> bool
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let n = a.nrows();
    for i in 0..n {
        for j in 0..n {
            if i != j && a[[i, j]].abs() > F::epsilon() {
                return false;
            }
        }
    }
    true
}

/// Check if a matrix is symmetric
pub fn is_symmetric<F>(a: &ArrayView2<F>) -> bool
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let n = a.nrows();
    if n != a.ncols() {
        return false;
    }

    for i in 0..n {
        for j in 0..n {
            if (a[[i, j]] - a[[j, i]]).abs() > F::epsilon() {
                return false;
            }
        }
    }
    true
}

/// Check if a matrix is the zero matrix
pub fn is_zero_matrix<F>(a: &ArrayView2<F>) -> bool
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let (m, n) = a.dim();
    for i in 0..m {
        for j in 0..n {
            if a[[i, j]].abs() > F::epsilon() {
                return false;
            }
        }
    }
    true
}

/// Check if a matrix is the identity matrix
pub fn is_identity<F>(a: &ArrayView2<F>) -> bool
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let n = a.nrows();
    if n != a.ncols() {
        return false;
    }

    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { F::one() } else { F::zero() };
            if (a[[i, j]] - expected).abs() > F::epsilon() {
                return false;
            }
        }
    }
    true
}

/// Compute matrix multiplication C = A * B
pub fn matrix_multiply<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions incompatible for multiplication: ({}, {}) × ({}, {})",
            m, k1, k2, n
        )));
    }

    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            for k in 0..k1 {
                c[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }

    Ok(c)
}

/// Compute matrix power for small integer powers using repeated squaring
pub fn integer_matrix_power<F>(a: &ArrayView2<F>, p: i32) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    use crate::solve::solve_multiple;

    let n = a.nrows();

    if p == 0 {
        return Ok(Array2::eye(n));
    }

    if p == 1 {
        return Ok(a.to_owned());
    }

    if p < 0 {
        // Compute A^{-|p|} = (A^{-1})^{|p|}
        let a_inv = solve_multiple(a, &Array2::eye(n).view(), None)?;
        return integer_matrix_power(&a_inv.view(), -p);
    }

    // Use repeated squaring for positive powers
    let mut result = Array2::eye(n);
    let mut base = a.to_owned();
    let mut exp = p as u32;

    while exp > 0 {
        if exp % 2 == 1 {
            result = matrix_multiply(&result.view(), &base.view())?;
        }
        base = matrix_multiply(&base.view(), &base.view())?;
        exp /= 2;
    }

    Ok(result)
}

/// Compute the Frobenius norm of a matrix
pub fn frobenius_norm<F>(a: &ArrayView2<F>) -> F
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let (m, n) = a.dim();
    let mut sum = F::zero();

    for i in 0..m {
        for j in 0..n {
            sum += a[[i, j]] * a[[i, j]];
        }
    }

    sum.sqrt()
}

/// Compute the maximum absolute difference between two matrices
pub fn matrix_diff_norm<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    if a.dim() != b.dim() {
        return Err(LinalgError::ShapeError(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let (m, n) = a.dim();
    let mut max_diff = F::zero();

    for i in 0..m {
        for j in 0..n {
            let diff = (a[[i, j]] - b[[i, j]]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    Ok(max_diff)
}

/// Scale a matrix by a scalar: result = alpha * A
pub fn scale_matrix<F>(a: &ArrayView2<F>, alpha: F) -> Array2<F>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let (m, n) = a.dim();
    let mut result = Array2::<F>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = alpha * a[[i, j]];
        }
    }

    result
}

/// Add two matrices: result = A + B
pub fn matrix_add<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    if a.dim() != b.dim() {
        return Err(LinalgError::ShapeError(
            "Matrices must have the same dimensions for addition".to_string(),
        ));
    }

    let (m, n) = a.dim();
    let mut result = Array2::<F>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = a[[i, j]] + b[[i, j]];
        }
    }

    Ok(result)
}

/// Subtract two matrices: result = A - B
pub fn matrix_subtract<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    if a.dim() != b.dim() {
        return Err(LinalgError::ShapeError(
            "Matrices must have the same dimensions for subtraction".to_string(),
        ));
    }

    let (m, n) = a.dim();
    let mut result = Array2::<F>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = a[[i, j]] - b[[i, j]];
        }
    }

    Ok(result)
}

/// Compute the matrix transpose
pub fn matrix_transpose<F>(a: &ArrayView2<F>) -> Array2<F>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let (m, n) = a.dim();
    let mut result = Array2::<F>::zeros((n, m));

    for i in 0..m {
        for j in 0..n {
            result[[j, i]] = a[[i, j]];
        }
    }

    result
}

/// Check if all eigenvalues are positive (for positive definite matrices)
pub fn check_positive_definite<F>(eigenvals: &[F]) -> bool
where
    F: Float,
{
    eigenvals.iter().all(|&val| val > F::zero())
}

/// Check if all eigenvalues are non-negative (for positive semidefinite matrices)
pub fn check_positive_semidefinite<F>(eigenvals: &[F]) -> bool
where
    F: Float,
{
    eigenvals.iter().all(|&val| val >= F::zero())
}
