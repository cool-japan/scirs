//! Matrix hyperbolic functions

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, One};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::validation::validate_decomposition;

/// Compute the matrix hyperbolic cosine.
///
/// The matrix hyperbolic cosine is computed using:
/// cosh(A) = (exp(A) + exp(-A)) / 2
///
/// For efficiency, this can also be computed using the series expansion:
/// cosh(A) = I + A²/2! + A⁴/4! + A⁶/6! + ...
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix hyperbolic cosine of a
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::coshm;
///
/// let a = array![[0.0_f64, 1.0], [1.0, 0.0]];
/// let cosh_a = coshm(&a.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn coshm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    validate_decomposition(a, "Matrix hyperbolic cosine computation", true)?;

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
        return Ok(Array2::eye(n)); // cosh(0) = I
    }

    // Special case for diagonal matrix
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
            result[[i, i]] = a[[i, i]].cosh();
        }
        return Ok(result);
    }

    // For general matrices, use series expansion
    // cosh(A) = I + A²/2! + A⁴/4! + A⁶/6! + ...

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

    // Compute cosh(A) = I + A²/2! + A⁴/4! + A⁶/6! + ...
    let mut result = Array2::eye(n);

    // Add A²/2!
    let two_factorial = F::from(2.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] += a2[[i, j]] / two_factorial;
        }
    }

    // Add A⁴/4!
    let four_factorial = F::from(24.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] += a4[[i, j]] / four_factorial;
        }
    }

    // Add A⁶/6!
    let six_factorial = F::from(720.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] += a6[[i, j]] / six_factorial;
        }
    }

    Ok(result)
}

/// Compute the matrix hyperbolic sine.
///
/// The matrix hyperbolic sine is computed using:
/// sinh(A) = (exp(A) - exp(-A)) / 2
///
/// For efficiency, this can also be computed using the series expansion:
/// sinh(A) = A + A³/3! + A⁵/5! + A⁷/7! + ...
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix hyperbolic sine of a
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::sinhm;
///
/// let a = array![[0.0_f64, 1.0], [1.0, 0.0]];
/// let sinh_a = sinhm(&a.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn sinhm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    validate_decomposition(a, "Matrix hyperbolic sine computation", true)?;

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
        return Ok(Array2::<F>::zeros((n, n))); // sinh(0) = 0
    }

    // Special case for diagonal matrix
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
            result[[i, i]] = a[[i, i]].sinh();
        }
        return Ok(result);
    }

    // For general matrices, use series expansion
    // sinh(A) = A + A³/3! + A⁵/5! + A⁷/7! + ...

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

    // Compute sinh(A) = A + A³/3! + A⁵/5! + A⁷/7! + ...
    let mut result = a.to_owned();

    // Add A³/3!
    let three_factorial = F::from(6.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] += a3[[i, j]] / three_factorial;
        }
    }

    // Add A⁵/5!
    let five_factorial = F::from(120.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] += a5[[i, j]] / five_factorial;
        }
    }

    // Add A⁷/7!
    let seven_factorial = F::from(5040.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] += a7[[i, j]] / seven_factorial;
        }
    }

    Ok(result)
}

/// Compute the matrix hyperbolic tangent.
///
/// The matrix hyperbolic tangent is computed as tanh(A) = sinh(A) * cosh(A)^{-1}
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix hyperbolic tangent of a
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::tanhm;
///
/// let a = array![[0.1_f64, 0.0], [0.0, 0.1]];
/// let tanh_a = tanhm(&a.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn tanhm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    use crate::solve::solve_multiple;

    let sinh_a = sinhm(a)?;
    let cosh_a = coshm(a)?;

    // Compute tanh(A) = sinh(A) * cosh(A)^{-1}
    // which is equivalent to solving cosh(A) * X = sinh(A)
    solve_multiple(&cosh_a.view(), &sinh_a.view(), None)
}
