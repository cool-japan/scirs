//! Basic matrix operations

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumAssign};
use std::iter::Sum;

/// Compute the determinant of a square matrix.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Determinant of the matrix
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::det;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let d = det(&a.view()).unwrap();
/// assert!((d - (-2.0)).abs() < 1e-10);
/// ```
pub fn det<F>(a: &ArrayView2<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix must be square to compute determinant, got shape {:?}",
            a.shape()
        )));
    }

    // Simple implementation for 2x2 and 3x3 matrices
    match a.nrows() {
        0 => Ok(F::one()),
        1 => Ok(a[[0, 0]]),
        2 => Ok(a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]),
        3 => {
            let det = a[[0, 0]] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
                - a[[0, 1]] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
                + a[[0, 2]] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]);
            Ok(det)
        }
        _ => {
            // For larger matrices, use LU decomposition
            use crate::decomposition::lu;

            match lu(a) {
                Ok((p, _l, u)) => {
                    // Calculate the determinant as the product of diagonal elements of U
                    let mut det_u = F::one();
                    for i in 0..u.nrows() {
                        det_u *= u[[i, i]];
                    }

                    // Count the number of row swaps in the permutation matrix
                    let mut swap_count = 0;
                    for i in 0..p.nrows() {
                        for j in 0..i {
                            if p[[i, j]] == F::one() {
                                swap_count += 1;
                            }
                        }
                    }

                    // Determinant is (-1)^swaps * det(U)
                    if swap_count % 2 == 0 {
                        Ok(det_u)
                    } else {
                        Ok(-det_u)
                    }
                }
                Err(LinalgError::SingularMatrixError(_)) => {
                    // Singular matrix has determinant zero
                    Ok(F::zero())
                }
                Err(e) => Err(e),
            }
        }
    }
}

/// Compute the inverse of a square matrix.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Inverse of the matrix
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::inv;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let a_inv = inv(&a.view()).unwrap();
/// assert!((a_inv[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((a_inv[[1, 1]] - 0.5).abs() < 1e-10);
/// ```
pub fn inv<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix must be square to compute inverse, got shape {:?}",
            a.shape()
        )));
    }

    // Simple implementation for 2x2 matrices
    if a.nrows() == 2 {
        let det_val = det(a)?;
        if det_val.abs() < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(format!(
                "Matrix is singular, cannot compute inverse\nMatrix shape: {:?}\nHint: Check if the matrix is singular or nearly singular",
                a.shape()
            )));
        }

        let inv_det = F::one() / det_val;
        let mut result = Array2::zeros((2, 2));
        result[[0, 0]] = a[[1, 1]] * inv_det;
        result[[0, 1]] = -a[[0, 1]] * inv_det;
        result[[1, 0]] = -a[[1, 0]] * inv_det;
        result[[1, 1]] = a[[0, 0]] * inv_det;
        return Ok(result);
    }

    // For larger matrices, use the solve_multiple function with an identity matrix
    use crate::solve::solve_multiple;

    let n = a.nrows();
    let mut identity = Array2::zeros((n, n));
    for i in 0..n {
        identity[[i, i]] = F::one();
    }

    // Solve A * X = I to get X = A^(-1)
    match solve_multiple(a, &identity.view()) {
        Err(LinalgError::SingularMatrixError(msg)) => {
            // Add diagnostic information to error message
            let diagnostic_msg = format!(
                "{}\nMatrix shape: {:?}\nHint: Check if the matrix is singular or nearly singular",
                msg,
                a.shape()
            );
            Err(LinalgError::SingularMatrixError(diagnostic_msg))
        }
        other => other,
    }
}

/// Raise a square matrix to the given power.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `n` - Power (can be positive, negative, or zero)
///
/// # Returns
///
/// * Matrix raised to the power n
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_power;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
///
/// // Identity matrix for n=0
/// let a_0 = matrix_power(&a.view(), 0).unwrap();
/// assert!((a_0[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((a_0[[0, 1]] - 0.0).abs() < 1e-10);
/// assert!((a_0[[1, 0]] - 0.0).abs() < 1e-10);
/// assert!((a_0[[1, 1]] - 1.0).abs() < 1e-10);
/// ```
pub fn matrix_power<F>(a: &ArrayView2<F>, n: i32) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix must be square to compute power, got shape {:?}",
            a.shape()
        )));
    }

    let dim = a.nrows();

    // Handle special cases
    if n == 0 {
        // Return identity matrix
        let mut result = Array2::zeros((dim, dim));
        for i in 0..dim {
            result[[i, i]] = F::one();
        }
        return Ok(result);
    }

    if n == 1 {
        // Return copy of the matrix
        return Ok(a.to_owned());
    }

    if n == -1 {
        // Return inverse
        return inv(a);
    }

    if n.abs() > 1 {
        // For higher powers, we would implement more efficient algorithms
        // using matrix decompositions or binary exponentiation
        // This is a placeholder that will be replaced with a proper implementation
        return Err(LinalgError::NotImplementedError(
            "Matrix power for |n| > 1 not yet implemented".to_string(),
        ));
    }

    // This should never be reached
    Err(LinalgError::ComputationError(
        "Unexpected error in matrix power calculation".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_det_2x2() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let d = det(&a.view()).unwrap();
        assert!((d - (-2.0)).abs() < 1e-10);

        let b = array![[2.0, 0.0], [0.0, 3.0]];
        let d = det(&b.view()).unwrap();
        assert!((d - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_det_3x3() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let d = det(&a.view()).unwrap();
        assert!((d - 0.0).abs() < 1e-10);

        let b = array![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let d = det(&b.view()).unwrap();
        assert!((d - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_inv_2x2() {
        let a = array![[1.0, 0.0], [0.0, 2.0]];
        let a_inv = inv(&a.view()).unwrap();
        assert_relative_eq!(a_inv[[0, 0]], 1.0);
        assert_relative_eq!(a_inv[[0, 1]], 0.0);
        assert_relative_eq!(a_inv[[1, 0]], 0.0);
        assert_relative_eq!(a_inv[[1, 1]], 0.5);

        let b = array![[1.0, 2.0], [3.0, 4.0]];
        let b_inv = inv(&b.view()).unwrap();
        assert_relative_eq!(b_inv[[0, 0]], -2.0);
        assert_relative_eq!(b_inv[[0, 1]], 1.0);
        assert_relative_eq!(b_inv[[1, 0]], 1.5);
        assert_relative_eq!(b_inv[[1, 1]], -0.5);
    }

    #[test]
    fn test_inv_large() {
        // Test 3x3 matrix
        let a = array![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let a_inv = inv(&a.view()).unwrap();

        // Verify A * A^(-1) = I
        let product = a.dot(&a_inv);
        let n = a.nrows();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_relative_eq!(product[[i, j]], 1.0, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(product[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }

        // Test 4x4 diagonal matrix
        let b = array![
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 5.0]
        ];
        let b_inv = inv(&b.view()).unwrap();
        assert_relative_eq!(b_inv[[0, 0]], 0.5, epsilon = 1e-10);
        assert_relative_eq!(b_inv[[1, 1]], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(b_inv[[2, 2]], 0.25, epsilon = 1e-10);
        assert_relative_eq!(b_inv[[3, 3]], 0.2, epsilon = 1e-10);

        // Test singular matrix should error
        let c = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]];
        assert!(inv(&c.view()).is_err());
    }

    #[test]
    fn test_matrix_power() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];

        // Power 0 should give identity matrix
        let a_0 = matrix_power(&a.view(), 0).unwrap();
        assert_relative_eq!(a_0[[0, 0]], 1.0);
        assert_relative_eq!(a_0[[0, 1]], 0.0);
        assert_relative_eq!(a_0[[1, 0]], 0.0);
        assert_relative_eq!(a_0[[1, 1]], 1.0);

        // Power 1 should return the original matrix
        let a_1 = matrix_power(&a.view(), 1).unwrap();
        assert_relative_eq!(a_1[[0, 0]], a[[0, 0]]);
        assert_relative_eq!(a_1[[0, 1]], a[[0, 1]]);
        assert_relative_eq!(a_1[[1, 0]], a[[1, 0]]);
        assert_relative_eq!(a_1[[1, 1]], a[[1, 1]]);
    }

    #[test]
    fn test_det_large() {
        // Test 4x4 matrix
        let a = array![
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0]
        ];
        let d = det(&a.view()).unwrap();
        assert_relative_eq!(d, 5.0, epsilon = 1e-10);

        // Test 5x5 diagonal matrix
        let b = array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 5.0]
        ];
        let d = det(&b.view()).unwrap();
        assert_relative_eq!(d, 120.0, epsilon = 1e-10);

        // Test singular matrix
        let c = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 6.0, 9.0, 12.0],
            [4.0, 8.0, 12.0, 16.0]
        ];
        let d = det(&c.view()).unwrap();
        assert_relative_eq!(d, 0.0, epsilon = 1e-10);
    }
}
