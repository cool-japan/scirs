//! Special matrix functions including sigmoid, softmax, and sign functions

use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumAssign, One};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::validation::validate_decomposition;

/// Compute the softmax function along a specified axis of a matrix.
///
/// The softmax function is defined as:
/// softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `axis` - Axis along which to compute softmax (None for element-wise)
///
/// # Returns
///
/// * Matrix with softmax applied
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::softmax;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let soft_a = softmax(&a.view(), Some(1)).unwrap();
/// ```
#[allow(dead_code)]
pub fn softmax<F>(a: &ArrayView2<F>, axis: Option<usize>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    let (nrows, ncols) = a.dim();

    if nrows == 0 || ncols == 0 {
        return Err(LinalgError::ShapeError(
            "Matrix must be non-empty for softmax computation".to_string(),
        ));
    }

    let mut result = Array2::<F>::zeros((nrows, ncols));

    match axis {
        None => {
            // Element-wise softmax (treat as flattened)
            let mut max_val = a[[0, 0]];
            for i in 0..nrows {
                for j in 0..ncols {
                    if a[[i, j]] > max_val {
                        max_val = a[[i, j]];
                    }
                }
            }

            // Compute exp(x - max) for numerical stability
            let mut sum = F::zero();
            for i in 0..nrows {
                for j in 0..ncols {
                    let exp_val = (a[[i, j]] - max_val).exp();
                    result[[i, j]] = exp_val;
                    sum += exp_val;
                }
            }

            // Normalize by sum
            for i in 0..nrows {
                for j in 0..ncols {
                    result[[i, j]] /= sum;
                }
            }
        }
        Some(0) => {
            // Softmax along rows (each column is normalized)
            for j in 0..ncols {
                // Find max in column j
                let mut max_val = a[[0, j]];
                for i in 1..nrows {
                    if a[[i, j]] > max_val {
                        max_val = a[[i, j]];
                    }
                }

                // Compute exp(x - max) and sum
                let mut sum = F::zero();
                for i in 0..nrows {
                    let exp_val = (a[[i, j]] - max_val).exp();
                    result[[i, j]] = exp_val;
                    sum += exp_val;
                }

                // Normalize
                for i in 0..nrows {
                    result[[i, j]] /= sum;
                }
            }
        }
        Some(1) => {
            // Softmax along columns (each row is normalized)
            for i in 0..nrows {
                // Find max in row i
                let mut max_val = a[[i, 0]];
                for j in 1..ncols {
                    if a[[i, j]] > max_val {
                        max_val = a[[i, j]];
                    }
                }

                // Compute exp(x - max) and sum
                let mut sum = F::zero();
                for j in 0..ncols {
                    let exp_val = (a[[i, j]] - max_val).exp();
                    result[[i, j]] = exp_val;
                    sum += exp_val;
                }

                // Normalize
                for j in 0..ncols {
                    result[[i, j]] /= sum;
                }
            }
        }
        Some(axis_val) => {
            return Err(LinalgError::InvalidInputError(format!(
                "Invalid axis {} for 2D matrix. Must be 0, 1, or None",
                axis_val
            )));
        }
    }

    Ok(result)
}

/// Compute the sigmoid function element-wise on a matrix.
///
/// The sigmoid function is defined as:
/// sigmoid(x) = 1 / (1 + exp(-x))
///
/// # Arguments
///
/// * `a` - Input matrix
///
/// # Returns
///
/// * Matrix with sigmoid applied element-wise
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::sigmoid;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let sig_a = sigmoid(&a.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn sigmoid<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    let (nrows, ncols) = a.dim();
    let mut result = Array2::<F>::zeros((nrows, ncols));

    for i in 0..nrows {
        for j in 0..ncols {
            let x = a[[i, j]];
            // Use numerically stable version: 1 / (1 + exp(-x))
            // For very negative x, this becomes approximately exp(x)
            // For very positive x, this stays close to 1
            if x > F::zero() {
                let exp_neg_x = (-x).exp();
                result[[i, j]] = F::one() / (F::one() + exp_neg_x);
            } else {
                let exp_x = x.exp();
                result[[i, j]] = exp_x / (F::one() + exp_x);
            }
        }
    }

    Ok(result)
}

/// Compute the matrix sign function.
///
/// The matrix sign function is defined as:
/// sign(A) = A * (A²)^{-1/2}
///
/// For eigendecomposition A = VDV^{-1}, this becomes:
/// sign(A) = V * sign(D) * V^{-1}
///
/// where sign(D) is the diagonal matrix with sign(λ_i) on the diagonal.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix sign of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::signm;
///
/// let a = array![[2.0_f64, 0.0], [0.0, -3.0]];
/// let sign_a = signm(&a.view()).unwrap();
/// // Should be approximately [[1.0, 0.0], [0.0, -1.0]]
/// ```
#[allow(dead_code)]
pub fn signm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    use crate::eigen::eig;
    use crate::solve::solve_multiple;

    validate_decomposition(a, "Matrix sign computation", true)?;

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
        return Ok(Array2::<F>::zeros((n, n))); // sign(0) = 0
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
            let val = a[[i, i]];
            if val > F::zero() {
                result[[i, i]] = F::one();
            } else if val < F::zero() {
                result[[i, i]] = -F::one();
            } else {
                result[[i, i]] = F::zero();
            }
        }
        return Ok(result);
    }

    // For general matrices, return a simplified error for now
    // A full implementation would require complex eigenvalue handling
    Err(LinalgError::ImplementationError(
        "Matrix sign function for general matrices is not yet fully implemented".to_string(),
    ))
}
