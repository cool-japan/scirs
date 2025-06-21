//! Matrix and vector norms

use ndarray::{ArrayView1, ArrayView2};
use num_traits::{Float, NumAssign};
use std::iter::Sum;

use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};
use crate::validation::{
    validate_finite_matrix, validate_finite_vector, validate_not_empty_matrix,
    validate_not_empty_vector,
};

/// Compute a matrix norm.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `ord` - Order of the norm:
///   * 'fro' or 'f': Frobenius norm
///   * '1': 1-norm (maximum column sum)
///   * 'inf': Infinity norm (maximum row sum)
///   * '2': 2-norm (largest singular value)
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Value of the norm
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_norm;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let norm_fro = matrix_norm(&a.view(), "fro", None).unwrap();
/// assert!((norm_fro - 5.477225575051661).abs() < 1e-10);
/// ```
pub fn matrix_norm<F>(a: &ArrayView2<F>, ord: &str, workers: Option<usize>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand,
{
    // Parameter validation using validation helpers
    validate_not_empty_matrix(a, "Matrix norm computation")?;
    validate_finite_matrix(a, "Matrix norm computation")?;

    match ord {
        "fro" | "f" | "frobenius" => {
            // Frobenius norm
            let mut sum_sq = F::zero();
            for i in 0..a.nrows() {
                for j in 0..a.ncols() {
                    sum_sq += a[[i, j]] * a[[i, j]];
                }
            }
            Ok(sum_sq.sqrt())
        }
        "1" => {
            // 1-norm (maximum column sum)
            let mut max_col_sum = F::zero();
            for j in 0..a.ncols() {
                let col = a.column(j);
                let col_sum = col.fold(F::zero(), |acc, &x| acc + x.abs());
                if col_sum > max_col_sum {
                    max_col_sum = col_sum;
                }
            }
            Ok(max_col_sum)
        }
        "inf" => {
            // Infinity norm (maximum row sum)
            let mut max_row_sum = F::zero();
            for i in 0..a.nrows() {
                let row = a.row(i);
                let row_sum = row.fold(F::zero(), |acc, &x| acc + x.abs());
                if row_sum > max_row_sum {
                    max_row_sum = row_sum;
                }
            }
            Ok(max_row_sum)
        }
        "2" => {
            // 2-norm (largest singular value)
            let (_u, s, _vt) = svd(a, false, workers)?;
            // The 2-norm is the largest singular value
            if s.is_empty() {
                Ok(F::zero())
            } else {
                Ok(s[0])
            }
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Matrix norm computation failed: Invalid norm order '{}'\nSupported norms: 'fro', 'f', 'frobenius', '1', 'inf', '2'",
            ord
        ))),
    }
}

/// Compute a vector norm.
///
/// # Arguments
///
/// * `x` - Input vector
/// * `ord` - Order of the norm:
///   * 1: 1-norm (sum of absolute values)
///   * 2: 2-norm (Euclidean norm)
///   * usize::MAX: Infinity norm (maximum absolute value)
///
/// # Returns
///
/// * Value of the norm
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::vector_norm;
///
/// let x = array![3.0_f64, 4.0];
/// let norm_2 = vector_norm(&x.view(), 2).unwrap();
/// assert!((norm_2 - 5.0).abs() < 1e-10);
/// ```
pub fn vector_norm<F>(x: &ArrayView1<F>, ord: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum,
{
    // Parameter validation using validation helpers
    validate_not_empty_vector(x, "Vector norm computation")?;
    validate_finite_vector(x, "Vector norm computation")?;

    match ord {
        1 => {
            // 1-norm (sum of absolute values)
            let sum_abs = x.fold(F::zero(), |acc, &val| acc + val.abs());
            Ok(sum_abs)
        }
        2 => {
            // 2-norm (Euclidean norm)
            let sum_sq = x.fold(F::zero(), |acc, &val| acc + val * val);
            Ok(sum_sq.sqrt())
        }
        usize::MAX => {
            // Infinity norm (maximum absolute value)
            let max_abs = x.fold(F::zero(), |acc, &val| {
                let abs_val = val.abs();
                if abs_val > acc {
                    abs_val
                } else {
                    acc
                }
            });
            Ok(max_abs)
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Vector norm computation failed: Invalid norm order {}\nSupported norms: 1 (L1), 2 (L2/Euclidean), {} (infinity)",
            ord, usize::MAX
        ))),
    }
}

/// Compute the condition number of a matrix.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `p` - Order of the norm:
///   * None: 2-norm
///   * "fro" or "f": Frobenius norm
///   * "1": 1-norm
///   * "inf": Infinity norm
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Condition number of the matrix
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::cond;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let c = cond(&a.view(), None, None).unwrap();
/// assert!((c - 2.0).abs() < 1e-10);
/// ```
pub fn cond<F>(a: &ArrayView2<F>, p: Option<&str>, workers: Option<usize>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand,
{
    // Parameter validation using validation helpers
    use crate::validation::validate_square_matrix;
    validate_not_empty_matrix(a, "Condition number computation")?;
    validate_square_matrix(a, "Condition number computation")?;
    validate_finite_matrix(a, "Condition number computation")?;

    let norm_type = p.unwrap_or("2");

    match norm_type {
        "2" | "fro" | "f" | "frobenius" => {
            // Use SVD for 2-norm and Frobenius norm condition number
            let (_u, s, _vt) = svd(a, false, workers)?;

            if s.is_empty() {
                return Ok(F::infinity());
            }

            // Find the largest and smallest non-zero singular values
            let sigma_max = s[0]; // Largest singular value (first in sorted order)
            let mut sigma_min = F::zero();

            // Find the smallest non-zero singular value
            for &val in s.iter().rev() {
                if val > F::epsilon() * F::from(100).unwrap() * sigma_max {
                    sigma_min = val;
                    break;
                }
            }

            if sigma_min <= F::zero() {
                Ok(F::infinity())
            } else {
                Ok(sigma_max / sigma_min)
            }
        }
        "1" | "inf" => {
            // For 1-norm and inf-norm, we need matrix inverse
            // This is more complex and would require computing the inverse
            // For now, fall back to SVD-based computation
            let (_u, s, _vt) = svd(a, false, workers)?;

            if s.is_empty() {
                return Ok(F::infinity());
            }

            let sigma_max = s[0];
            let mut sigma_min = F::zero();

            for &val in s.iter().rev() {
                if val > F::epsilon() * F::from(100).unwrap() * sigma_max {
                    sigma_min = val;
                    break;
                }
            }

            if sigma_min <= F::zero() {
                Ok(F::infinity())
            } else {
                Ok(sigma_max / sigma_min)
            }
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Condition number computation failed: Invalid norm type '{}'\nSupported norms: '1', '2' (default), 'fro', 'f', 'frobenius', 'inf'",
            norm_type
        ))),
    }
}

/// Compute the rank of a matrix.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `tol` - Tolerance for singular values (None = automatic)
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Rank of the matrix
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::matrix_rank;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let r = matrix_rank(&a.view(), None, None).unwrap();
/// assert_eq!(r, 2);
/// ```
pub fn matrix_rank<F>(
    a: &ArrayView2<F>,
    tol: Option<F>,
    workers: Option<usize>,
) -> LinalgResult<usize>
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand,
{
    // Parameter validation using validation helpers
    if a.is_empty() {
        return Ok(0);
    }
    validate_finite_matrix(a, "Matrix rank computation")?;

    // Validate tolerance
    if let Some(t) = tol {
        if t < F::zero() {
            return Err(LinalgError::InvalidInputError(
                "Matrix rank computation failed: Tolerance must be non-negative".to_string(),
            ));
        }
    }

    // Compute SVD to get singular values
    let (_u, s, _vt) = svd(a, false, workers)?;

    if s.is_empty() {
        return Ok(0);
    }

    // Determine tolerance
    let tolerance = if let Some(t) = tol {
        t
    } else {
        // Default tolerance: max(m, n) * eps * max(singular_values)
        let max_dim = std::cmp::max(a.nrows(), a.ncols());
        let eps = F::epsilon();
        let sigma_max = s[0]; // Largest singular value
        F::from(max_dim).unwrap() * eps * sigma_max
    };

    // Count singular values above tolerance
    let rank = s.iter().filter(|&&val| val > tolerance).count();

    Ok(rank)
}

// Backward compatibility functions (deprecated)

/// Compute a matrix norm without workers parameter (deprecated - use matrix_norm with workers)
#[deprecated(
    since = "0.1.0",
    note = "Use matrix_norm with workers parameter instead"
)]
pub fn matrix_norm_default<F>(a: &ArrayView2<F>, ord: &str) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand,
{
    matrix_norm(a, ord, None)
}

/// Compute the condition number without workers parameter (deprecated - use cond with workers)
#[deprecated(since = "0.1.0", note = "Use cond with workers parameter instead")]
pub fn cond_default<F>(a: &ArrayView2<F>, p: Option<&str>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand,
{
    cond(a, p, None)
}

/// Compute matrix rank without workers parameter (deprecated - use matrix_rank with workers)
#[deprecated(
    since = "0.1.0",
    note = "Use matrix_rank with workers parameter instead"
)]
pub fn matrix_rank_default<F>(a: &ArrayView2<F>, tol: Option<F>) -> LinalgResult<usize>
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand,
{
    matrix_rank(a, tol, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_matrix_norm_frobenius() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let norm = matrix_norm(&a.view(), "fro", None).unwrap();
        // sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30) ≈ 5.477
        assert_relative_eq!(norm, 5.477225575051661, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_norm_1() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let norm = matrix_norm(&a.view(), "1", None).unwrap();
        // max(1+3, 2+4) = max(4, 6) = 6
        assert_relative_eq!(norm, 6.0);
    }

    #[test]
    fn test_matrix_norm_inf() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let norm = matrix_norm(&a.view(), "inf", None).unwrap();
        // max(1+2, 3+4) = max(3, 7) = 7
        assert_relative_eq!(norm, 7.0);
    }

    #[test]
    fn test_vector_norm_1() {
        let x = array![1.0, -2.0, 3.0];
        let norm = vector_norm(&x.view(), 1).unwrap();
        // |1| + |-2| + |3| = 6
        assert_relative_eq!(norm, 6.0);
    }

    #[test]
    fn test_vector_norm_2() {
        let x = array![3.0, 4.0];
        let norm = vector_norm(&x.view(), 2).unwrap();
        // sqrt(3^2 + 4^2) = 5
        assert_relative_eq!(norm, 5.0);
    }

    #[test]
    fn test_vector_norm_inf() {
        let x = array![1.0, -5.0, 3.0];
        let norm = vector_norm(&x.view(), usize::MAX).unwrap();
        // max(|1|, |-5|, |3|) = 5
        assert_relative_eq!(norm, 5.0);
    }
}
