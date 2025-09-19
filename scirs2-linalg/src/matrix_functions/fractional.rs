//! Fractional matrix functions and advanced matrix power computations

use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumAssign, One};
use std::iter::Sum;

use crate::eigen::eigh;
use crate::error::{LinalgError, LinalgResult};
use crate::solve::solve_multiple;
use crate::validation::validate_decomposition;

/// Compute the fractional matrix power A^p using various methods.
///
/// This function provides multiple algorithms for computing fractional
/// matrix powers, including eigendecomposition and Padé approximation methods.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `p` - Power (real number, can be fractional)
/// * `method` - Method to use ("eigen", "pade", "schur")
///
/// # Returns
///
/// * Matrix power A^p
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::fractionalmatrix_power;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let a_half = fractionalmatrix_power(&a.view(), 0.5, "eigen").unwrap();
/// // Should be approximately [[2.0, 0.0], [0.0, 3.0]]
/// ```
#[allow(dead_code)]
pub fn fractionalmatrix_power<F>(a: &ArrayView2<F>, p: F, method: &str) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + 'static + Send + Sync + ndarray::ScalarOperand,
{
    validate_decomposition(a, "Fractional matrix power computation", true)?;

    let n = a.nrows();

    // Special case for p = 0 (returns identity)
    if p.abs() < F::epsilon() {
        return Ok(Array2::eye(n));
    }

    // Special case for p = 1 (returns the matrix itself)
    if (p - F::one()).abs() < F::epsilon() {
        return Ok(a.to_owned());
    }

    match method {
        "eigen" => {
            // Use eigendecomposition method
            eigendecomposition_power(a, p)
        }
        "pade" => {
            // Use Padé approximation method
            pade_fractional_power(a, p)
        }
        "schur" => {
            // Use Schur decomposition method (simplified implementation)
            // For now, fall back to eigendecomposition
            eigendecomposition_power(a, p)
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Unknown method '{}'. Supported methods: 'eigen', 'pade', 'schur'",
            method
        ))),
    }
}

/// Compute matrix power using eigendecomposition.
fn eigendecomposition_power<F>(a: &ArrayView2<F>, p: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + 'static + Send + Sync + ndarray::ScalarOperand,
{
    let n = a.nrows();

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
            let val = a[[i, i]];
            if val < F::zero() && !is_integer(p) {
                return Err(LinalgError::InvalidInputError(
                    "Cannot compute real fractional power of negative number".to_string(),
                ));
            }
            result[[i, i]] = val.powf(p);
        }
        return Ok(result);
    }

    // For general matrices with non-integer powers, return a simplified error for now
    // A full implementation would require complex eigenvalue handling
    Err(LinalgError::ImplementationError(
        "Fractional matrix powers for general matrices are not yet fully implemented".to_string(),
    ))
}

/// Compute matrix power using Padé approximation.
fn pade_fractional_power<F>(a: &ArrayView2<F>, p: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + 'static + Send + Sync + ndarray::ScalarOperand,
{
    // For simplicity, this implementation uses a basic approach
    // A full Padé implementation would be more complex

    // If p is close to an integer, use repeated multiplication
    if is_integer(p) {
        return integer_power(a, p.to_i32().unwrap_or(0));
    }

    // For non-integer powers close to small rationals, use root extraction
    // This is a simplified implementation
    if (p - F::from(0.5).unwrap()).abs() < F::epsilon() {
        // Square root case
        use super::exponential::sqrtm;
        return sqrtm(a, 50, F::from(1e-12).unwrap());
    }

    // For other cases, fall back to eigendecomposition
    eigendecomposition_power(a, p)
}

/// Compute integer power using repeated squaring.
fn integer_power<F>(a: &ArrayView2<F>, p: i32) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + 'static + Send + Sync + ndarray::ScalarOperand,
{
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
        return integer_power(&a_inv.view(), -p);
    }

    // Use repeated squaring for positive powers
    let mut result = Array2::eye(n);
    let mut base = a.to_owned();
    let mut exp = p as u32;

    while exp > 0 {
        if exp % 2 == 1 {
            // Multiply result by base
            let mut temp = Array2::<F>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        temp[[i, j]] += result[[i, k]] * base[[k, j]];
                    }
                }
            }
            result = temp;
        }
        // Square the base
        let mut temp = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    temp[[i, j]] += base[[i, k]] * base[[k, j]];
                }
            }
        }
        base = temp;
        exp /= 2;
    }

    Ok(result)
}

/// Apply a general function to a symmetric positive definite matrix.
///
/// This function applies a scalar function f to a symmetric positive definite
/// matrix A using eigendecomposition: f(A) = V * f(D) * V^T
///
/// # Arguments
///
/// * `a` - Input symmetric positive definite matrix
/// * `f` - Function to apply (as a closure)
/// * `check_spd` - Whether to check that the matrix is SPD
///
/// # Returns
///
/// * f(A) where f is applied element-wise to the eigenvalues
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::spdmatrix_function;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let sqrt_a = spdmatrix_function(&a.view(), |x| x.sqrt(), true).unwrap();
/// ```
#[allow(dead_code)]
pub fn spdmatrix_function<F, Func>(
    a: &ArrayView2<F>,
    f: Func,
    check_spd: bool,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + 'static + Send + Sync + ndarray::ScalarOperand,
    Func: Fn(F) -> F,
{
    validate_decomposition(a, "SPD matrix function computation", true)?;

    let n = a.nrows();

    // Check if matrix is symmetric
    for i in 0..n {
        for j in 0..n {
            if (a[[i, j]] - a[[j, i]]).abs() > F::epsilon() {
                return Err(LinalgError::InvalidInputError(
                    "Matrix must be symmetric for SPD matrix function".to_string(),
                ));
            }
        }
    }

    // Check if matrix is positive definite if requested
    if check_spd {
        // Quick check: diagonal elements should be positive
        for i in 0..n {
            if a[[i, i]] <= F::zero() {
                return Err(LinalgError::InvalidInputError(
                    "Matrix must be positive definite".to_string(),
                ));
            }
        }
    }

    // For diagonal matrices, apply function directly to diagonal elements
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
        let mut result = Array2::zeros((n, n));
        for i in 0..n {
            result[[i, i]] = f(a[[i, i]]);
        }
        return Ok(result);
    }

    // For general SPD matrices, use eigendecomposition
    // Note: For real symmetric matrices, eigenvalues are real
    let (eigenvalues, eigenvectors) = eigh(a, None)?;

    // Apply function to eigenvalues
    let mut diag = Array2::zeros((n, n));
    for i in 0..n {
        let eigenval = eigenvalues[i];
        if check_spd && eigenval <= F::zero() {
            return Err(LinalgError::InvalidInputError(
                "Matrix is not positive definite (negative eigenvalue found)".to_string(),
            ));
        }
        diag[[i, i]] = f(eigenval);
    }

    // Reconstruct: A_f = V * diag(f(λ)) * V^T
    let temp = eigenvectors.dot(&diag);
    let v_t = eigenvectors.t();
    let result = temp.dot(&v_t);

    Ok(result)
}

/// Helper function to check if a floating point number is close to an integer
fn is_integer<F: Float>(x: F) -> bool {
    (x - x.round()).abs() < F::from(1e-10).unwrap_or(F::epsilon())
}
