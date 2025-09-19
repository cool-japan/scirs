//! Matrix trigonometric functions

use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumAssign, One};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::validation::validate_decomposition;

/// Compute the matrix cosine.
///
/// The matrix cosine is computed using the matrix exponential:
/// cos(A) = (exp(iA) + exp(-iA)) / 2
///
/// For real matrices, this is implemented using the series expansion:
/// cos(A) = I - A²/2! + A⁴/4! - A⁶/6! + ...
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix cosine of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::cosm;
///
/// let a = array![[0.0_f64, 1.0], [-1.0, 0.0]]; // Rotation matrix
/// let cos_a = cosm(&a.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn cosm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    validate_decomposition(a, "Matrix cosine computation", true)?;

    let n = a.nrows();

    // Special case for zero matrix
    let mut is_zero = true;
    for i in 0..n {
        for j in 0..n {
            if a[[i, j]].abs() > F::epsilon() {
                is_zero = false;
                break;
            }
        }
        if !is_zero {
            break;
        }
    }

    if is_zero {
        return Ok(Array2::eye(n)); // cos(0) = I
    }

    // For small matrices, use series expansion
    // cos(A) = I - A²/2! + A⁴/4! - A⁶/6! + ...

    // Compute powers of A
    let mut a2 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a2[[i, j]] += a[[i, k]] * a[[k, j]];
            }
        }
    }

    let mut a4 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a4[[i, j]] += a2[[i, k]] * a2[[k, j]];
            }
        }
    }

    let mut a6 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a6[[i, j]] += a4[[i, k]] * a2[[k, j]];
            }
        }
    }

    // Compute cos(A) = I - A²/2! + A⁴/4! - A⁶/6! + ...
    let mut result = Array2::eye(n);

    // Subtract A²/2!
    let two_factorial = F::from(2.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] -= a2[[i, j]] / two_factorial;
        }
    }

    // Add A⁴/4!
    let four_factorial = F::from(24.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] += a4[[i, j]] / four_factorial;
        }
    }

    // Subtract A⁶/6!
    let six_factorial = F::from(720.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] -= a6[[i, j]] / six_factorial;
        }
    }

    Ok(result)
}

/// Compute the matrix sine.
///
/// The matrix sine is computed using the matrix exponential:
/// sin(A) = (exp(iA) - exp(-iA)) / (2i)
///
/// For real matrices, this is implemented using the series expansion:
/// sin(A) = A - A³/3! + A⁵/5! - A⁷/7! + ...
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix sine of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::sinm;
///
/// let a = array![[0.0_f64, 1.0], [-1.0, 0.0]]; // Rotation matrix
/// let sin_a = sinm(&a.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn sinm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    validate_decomposition(a, "Matrix sine computation", true)?;

    let n = a.nrows();

    // Special case for zero matrix
    let mut is_zero = true;
    for i in 0..n {
        for j in 0..n {
            if a[[i, j]].abs() > F::epsilon() {
                is_zero = false;
                break;
            }
        }
        if !is_zero {
            break;
        }
    }

    if is_zero {
        return Ok(Array2::<F>::zeros((n, n))); // sin(0) = 0
    }

    // For small matrices, use series expansion
    // sin(A) = A - A³/3! + A⁵/5! - A⁷/7! + ...

    // Compute powers of A
    let mut a2 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a2[[i, j]] += a[[i, k]] * a[[k, j]];
            }
        }
    }

    let mut a3 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a3[[i, j]] += a2[[i, k]] * a[[k, j]];
            }
        }
    }

    let mut a5 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a5[[i, j]] += a3[[i, k]] * a2[[k, j]];
            }
        }
    }

    let mut a7 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a7[[i, j]] += a5[[i, k]] * a2[[k, j]];
            }
        }
    }

    // Compute sin(A) = A - A³/3! + A⁵/5! - A⁷/7! + ...
    let mut result = a.to_owned();

    // Subtract A³/3!
    let three_factorial = F::from(6.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] -= a3[[i, j]] / three_factorial;
        }
    }

    // Add A⁵/5!
    let five_factorial = F::from(120.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] += a5[[i, j]] / five_factorial;
        }
    }

    // Subtract A⁷/7!
    let seven_factorial = F::from(5040.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] -= a7[[i, j]] / seven_factorial;
        }
    }

    Ok(result)
}

/// Compute the matrix tangent.
///
/// The matrix tangent is computed as tan(A) = sin(A) * cos(A)^{-1}
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix tangent of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::tanm;
///
/// let a = array![[0.1_f64, 0.0], [0.0, 0.1]];
/// let tan_a = tanm(&a.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn tanm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    use crate::solve::solve_multiple;

    let sin_a = sinm(a)?;
    let cos_a = cosm(a)?;

    // Compute tan(A) = sin(A) * cos(A)^{-1}
    // which is equivalent to solving cos(A) * X = sin(A)
    solve_multiple(&cos_a.view(), &sin_a.view(), None)
}

/// Compute the matrix arccosine.
///
/// This function computes arccos(A) for a matrix A.
/// The implementation uses the identity: arccos(A) = -i * log(A + i * sqrt(I - A²))
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix arccosine of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::acosm;
///
/// let a = array![[0.5_f64, 0.0], [0.0, 0.5]];
/// let acos_a = acosm(&a.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn acosm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    validate_decomposition(a, "Matrix arccosine computation", true)?;

    // For simplicity, this is a basic implementation
    // A full implementation would use the complex logarithm approach

    let n = a.nrows();

    // Check if matrix is diagonal and use elementwise arccos
    let mut is_diagonal = true;
    for i in 0..n {
        for j in 0..n {
            if i != j && a[[i, j]].abs() > F::epsilon() {
                is_diagonal = false;
                break;
            }
        }
        if !is_diagonal {
            break;
        }
    }

    if is_diagonal {
        let mut result = Array2::<F>::zeros((n, n));
        for i in 0..n {
            let val = a[[i, i]];
            if val.abs() > F::one() {
                return Err(LinalgError::InvalidInputError(
                    "Matrix arccosine: diagonal elements must be in [-1, 1]".to_string(),
                ));
            }
            result[[i, i]] = val.acos();
        }
        return Ok(result);
    }

    // For non-diagonal matrices, return an error for now
    // A complete implementation would require complex matrix functions
    Err(LinalgError::ImplementationError(
        "Matrix arccosine for non-diagonal matrices is not yet implemented".to_string(),
    ))
}

/// Compute the matrix arcsine.
///
/// This function computes arcsin(A) for a matrix A.
/// The implementation uses the identity: arcsin(A) = -i * log(i * A + sqrt(I - A²))
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix arcsine of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::asinm;
///
/// let a = array![[0.5_f64, 0.0], [0.0, 0.5]];
/// let asin_a = asinm(&a.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn asinm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    validate_decomposition(a, "Matrix arcsine computation", true)?;

    // For simplicity, this is a basic implementation
    // A full implementation would use the complex logarithm approach

    let n = a.nrows();

    // Check if matrix is diagonal and use elementwise arcsin
    let mut is_diagonal = true;
    for i in 0..n {
        for j in 0..n {
            if i != j && a[[i, j]].abs() > F::epsilon() {
                is_diagonal = false;
                break;
            }
        }
        if !is_diagonal {
            break;
        }
    }

    if is_diagonal {
        let mut result = Array2::<F>::zeros((n, n));
        for i in 0..n {
            let val = a[[i, i]];
            if val.abs() > F::one() {
                return Err(LinalgError::InvalidInputError(
                    "Matrix arcsine: diagonal elements must be in [-1, 1]".to_string(),
                ));
            }
            result[[i, i]] = val.asin();
        }
        return Ok(result);
    }

    // For non-diagonal matrices, return an error for now
    // A complete implementation would require complex matrix functions
    Err(LinalgError::ImplementationError(
        "Matrix arcsine for non-diagonal matrices is not yet implemented".to_string(),
    ))
}

/// Compute the matrix arctangent.
///
/// This function computes arctan(A) for a matrix A.
/// The implementation uses the identity: arctan(A) = (i/2) * log((I + iA) / (I - iA))
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix arctangent of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::atanm;
///
/// let a = array![[0.5_f64, 0.0], [0.0, 0.5]];
/// let atan_a = atanm(&a.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn atanm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    validate_decomposition(a, "Matrix arctangent computation", true)?;

    // For simplicity, this is a basic implementation
    // A full implementation would use the complex logarithm approach

    let n = a.nrows();

    // Check if matrix is diagonal and use elementwise arctan
    let mut is_diagonal = true;
    for i in 0..n {
        for j in 0..n {
            if i != j && a[[i, j]].abs() > F::epsilon() {
                is_diagonal = false;
                break;
            }
        }
        if !is_diagonal {
            break;
        }
    }

    if is_diagonal {
        let mut result = Array2::<F>::zeros((n, n));
        for i in 0..n {
            result[[i, i]] = a[[i, i]].atan();
        }
        return Ok(result);
    }

    // For non-diagonal matrices, return an error for now
    // A complete implementation would require complex matrix functions
    Err(LinalgError::ImplementationError(
        "Matrix arctangent for non-diagonal matrices is not yet implemented".to_string(),
    ))
}
