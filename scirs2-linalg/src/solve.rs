//! Linear equation solvers

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumAssign, One};
use std::iter::Sum;

use crate::basic::inv;
use crate::decomposition::{lu, qr, svd};
use crate::error::{LinalgError, LinalgResult};
use crate::validation::{
    validate_finite_matrix, validate_finite_vector, validate_least_squares, validate_linear_system,
    validate_matrix_vector_dimensions, validate_multiple_linear_systems, validate_not_empty_matrix,
    validate_not_empty_vector, validate_square_matrix,
};

/// Solution to a least-squares problem
pub struct LstsqResult<F: Float> {
    /// Least-squares solution
    pub x: Array1<F>,
    /// Sum of squared residuals
    pub residuals: F,
    /// Rank of coefficient matrix
    pub rank: usize,
    /// Singular values
    pub s: Array1<F>,
}

/// Solve a linear system of equations.
///
/// Solves the equation a x = b for x, assuming a is a square matrix.
///
/// # Arguments
///
/// * `a` - Coefficient matrix
/// * `b` - Ordinate or "dependent variable" values
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Solution vector x
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::solve;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let b = array![2.0_f64, 3.0];
/// let x = solve(&a.view(), &b.view(), None).unwrap();
/// assert!((x[0] - 2.0).abs() < 1e-10);
/// assert!((x[1] - 3.0).abs() < 1e-10);
/// ```
pub fn solve<F>(
    a: &ArrayView2<F>,
    b: &ArrayView1<F>,
    workers: Option<usize>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + One + Sum,
{
    // Parameter validation using helper function
    validate_linear_system(a, b, "Linear system solve")?;

    // For small matrices, we can solve directly using the inverse
    if a.nrows() <= 4 {
        let a_inv = inv(a, None)?;
        // Compute x = a_inv * b
        let mut x = Array1::zeros(a.nrows());
        for i in 0..a.nrows() {
            for j in 0..a.nrows() {
                x[i] += a_inv[[i, j]] * b[j];
            }
        }
        return Ok(x);
    }

    // Configure OpenMP thread count if workers specified
    if let Some(num_workers) = workers {
        std::env::set_var("OMP_NUM_THREADS", num_workers.to_string());
    }

    // For larger systems, use LU decomposition
    let (p, l, u) = match lu(a, workers) {
        Err(LinalgError::SingularMatrixError(_)) => {
            return Err(LinalgError::singular_matrix_with_suggestions(
                "linear system solve",
                a.dim(),
                None,
            ))
        }
        Err(e) => return Err(e),
        Ok(result) => result,
    };

    // Compute P*b
    let mut pb = Array1::zeros(b.len());
    for i in 0..p.nrows() {
        for j in 0..p.ncols() {
            pb[i] += p[[i, j]] * b[j];
        }
    }

    // Solve L*y = P*b by forward substitution
    let y = solve_triangular(&l.view(), &pb.view(), true, true)?;

    // Solve U*x = y by back substitution
    let x = solve_triangular(&u.view(), &y.view(), false, false)?;

    Ok(x)
}

/// Solve a linear system with a lower or upper triangular coefficient matrix.
///
/// # Arguments
///
/// * `a` - Triangular coefficient matrix
/// * `b` - Ordinate or "dependent variable" values
/// * `lower` - If true, the matrix is lower triangular, if false, upper triangular
/// * `unit_diagonal` - If true, the diagonal elements of a are assumed to be 1
///
/// # Returns
///
/// * Solution vector x
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::solve_triangular;
///
/// // Lower triangular system
/// let a = array![[1.0_f64, 0.0], [2.0, 3.0]];
/// let b = array![2.0_f64, 8.0];
/// let x = solve_triangular(&a.view(), &b.view(), true, false).unwrap();
/// assert!((x[0] - 2.0).abs() < 1e-10);
/// assert!((x[1] - 4.0/3.0).abs() < 1e-10);
/// ```
pub fn solve_triangular<F>(
    a: &ArrayView2<F>,
    b: &ArrayView1<F>,
    lower: bool,
    unit_diagonal: bool,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum,
{
    // Parameter validation using helper functions
    validate_not_empty_matrix(a, "Triangular system solve")?;
    validate_not_empty_vector(b, "Triangular system solve")?;
    validate_square_matrix(a, "Triangular system solve")?;
    validate_matrix_vector_dimensions(a, b, "Triangular system solve")?;
    validate_finite_matrix(a, "Triangular system solve")?;
    validate_finite_vector(b, "Triangular system solve")?;

    let n = a.nrows();
    let mut x = Array1::zeros(n);

    if lower {
        // Forward substitution for lower triangular matrix
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= a[[i, j]] * x[j];
            }
            if unit_diagonal {
                x[i] = sum;
            } else {
                if a[[i, i]].abs() < F::epsilon() {
                    return Err(LinalgError::singular_matrix_with_suggestions(
                        "triangular system solve (forward substitution)",
                        a.dim(),
                        Some(1e16), // Very high condition number due to zero diagonal
                    ));
                }
                x[i] = sum / a[[i, i]];
            }
        }
    } else {
        // Back substitution for upper triangular matrix
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= a[[i, j]] * x[j];
            }
            if unit_diagonal {
                x[i] = sum;
            } else {
                if a[[i, i]].abs() < F::epsilon() {
                    return Err(LinalgError::singular_matrix_with_suggestions(
                        "triangular system solve (back substitution)",
                        a.dim(),
                        Some(1e16), // Very high condition number due to zero diagonal
                    ));
                }
                x[i] = sum / a[[i, i]];
            }
        }
    }

    Ok(x)
}

/// Compute least-squares solution to a linear matrix equation.
///
/// Computes the vector x that solves the least squares equation
/// a x = b by computing the full least squares solution.
///
/// # Arguments
///
/// * `a` - Coefficient matrix
/// * `b` - Ordinate or "dependent variable" values
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * A LstsqResult struct containing:
///   * x: Least-squares solution
///   * residuals: Sum of squared residuals
///   * rank: Rank of matrix a
///   * s: Singular values of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::lstsq;
///
/// let a = array![[1.0_f64, 1.0], [1.0, 2.0], [1.0, 3.0]];
/// let b = array![6.0_f64, 9.0, 12.0];
/// let result = lstsq(&a.view(), &b.view(), None).unwrap();
/// // result.x should be approximately [3.0, 3.0]
/// ```
pub fn lstsq<F>(
    a: &ArrayView2<F>,
    b: &ArrayView1<F>,
    workers: Option<usize>,
) -> LinalgResult<LstsqResult<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand,
{
    // Parameter validation using helper function
    validate_least_squares(a, b, "Least squares solve")?;

    // Configure OpenMP thread count if workers specified
    if let Some(num_workers) = workers {
        std::env::set_var("OMP_NUM_THREADS", num_workers.to_string());
    }

    // For underdetermined systems with full rank, use the normal equation approach
    if a.nrows() >= a.ncols() {
        // QR decomposition approach
        let (q, r) = qr(a, workers)?;

        // Compute Q^T * b
        let qt = q.t().to_owned();
        let mut qt_b = Array1::zeros(qt.nrows());
        for i in 0..qt.nrows() {
            for j in 0..qt.ncols() {
                qt_b[i] += qt[[i, j]] * b[j];
            }
        }

        // Get the effective rank
        let rank = a.ncols(); // Assume full rank for now

        // Extract the first part of Q^T * b corresponding to the rank
        let qt_b_truncated = qt_b.slice(ndarray::s![0..rank]).to_owned();

        // Solve R * x = Q^T * b using back substitution
        let r_truncated = r.slice(ndarray::s![0..rank, 0..a.ncols()]).to_owned();
        let x = solve_triangular(&r_truncated.view(), &qt_b_truncated.view(), false, false)?;

        // Compute residuals: ||Ax - b||²
        let mut residuals = F::zero();
        for i in 0..a.nrows() {
            let mut a_x_i = F::zero();
            for j in 0..a.ncols() {
                a_x_i += a[[i, j]] * x[j];
            }
            let diff = b[i] - a_x_i;
            residuals += diff * diff;
        }

        // Create singular values (empty for QR approach)
        let s = Array1::zeros(0);

        Ok(LstsqResult {
            x,
            residuals,
            rank,
            s,
        })
    } else {
        // Underdetermined system, use SVD
        let (u, s, vt) = svd(a, false, workers)?;

        // Determine effective rank by thresholding singular values
        let threshold = s[0] * F::from(a.nrows().max(a.ncols())).unwrap() * F::epsilon();
        let rank = s.iter().filter(|&&val| val > threshold).count();

        // Compute U^T * b
        let ut = u.t().to_owned();
        let mut ut_b = Array1::zeros(ut.nrows());
        for i in 0..ut.nrows() {
            for j in 0..ut.ncols() {
                ut_b[i] += ut[[i, j]] * b[j];
            }
        }

        // Initialize solution vector
        let mut x = Array1::zeros(a.ncols());

        // Solve using SVD components
        for i in 0..rank {
            let s_inv = F::one() / s[i];
            for j in 0..a.ncols() {
                x[j] += vt[[i, j]] * ut_b[i] * s_inv;
            }
        }

        // Compute residuals: ||Ax - b||²
        let mut residuals = F::zero();
        for i in 0..a.nrows() {
            let mut a_x_i = F::zero();
            for j in 0..a.ncols() {
                a_x_i += a[[i, j]] * x[j];
            }
            let diff = b[i] - a_x_i;
            residuals += diff * diff;
        }

        Ok(LstsqResult {
            x,
            residuals,
            rank,
            s,
        })
    }
}

/// Solve the linear system Ax = B for x with multiple right-hand sides.
///
/// # Arguments
///
/// * `a` - Coefficient matrix
/// * `b` - Matrix of right-hand sides where each column is a different right-hand side
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Solution matrix x where each column is a solution vector
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::solve_multiple;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let b = array![[2.0_f64, 4.0], [3.0, 5.0]];
/// let x = solve_multiple(&a.view(), &b.view(), None).unwrap();
/// // First column of x should be [2.0, 3.0]
/// // Second column of x should be [4.0, 5.0]
/// ```
pub fn solve_multiple<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    workers: Option<usize>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + One + Sum,
{
    // Parameter validation using helper function
    validate_multiple_linear_systems(a, b, "Multiple linear systems solve")?;

    // Configure OpenMP thread count if workers specified
    if let Some(num_workers) = workers {
        std::env::set_var("OMP_NUM_THREADS", num_workers.to_string());
    }

    // For efficiency, perform LU decomposition once
    let (p, l, u) = match lu(a, workers) {
        Err(LinalgError::SingularMatrixError(_)) => {
            return Err(LinalgError::singular_matrix_with_suggestions(
                "multiple linear systems solve",
                a.dim(),
                None,
            ))
        }
        Err(e) => return Err(e),
        Ok(result) => result,
    };

    // Initialize solution matrix
    let mut x = Array2::zeros((a.ncols(), b.ncols()));

    // Solve for each right-hand side
    for j in 0..b.ncols() {
        // Extract j-th right-hand side
        let b_j = b.column(j).to_owned();

        // Compute P*b
        let mut pb = Array1::zeros(b_j.len());
        for i in 0..p.nrows() {
            for k in 0..p.ncols() {
                pb[i] += p[[i, k]] * b_j[k];
            }
        }

        // Solve L*y = P*b by forward substitution
        let y = solve_triangular(&l.view(), &pb.view(), true, true)?;

        // Solve U*x = y by back substitution
        let x_j = solve_triangular(&u.view(), &y.view(), false, false)?;

        // Store solution in the j-th column of x
        for i in 0..x_j.len() {
            x[[i, j]] = x_j[i];
        }
    }

    Ok(x)
}

// Convenience wrapper functions for backward compatibility

/// Solve linear system using default thread count
pub fn solve_default<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + One + Sum,
{
    solve(a, b, None)
}

/// Compute least-squares solution using default thread count
pub fn lstsq_default<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> LinalgResult<LstsqResult<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand,
{
    lstsq(a, b, None)
}

/// Solve multiple linear systems using default thread count
pub fn solve_multiple_default<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + One + Sum,
{
    solve_multiple(a, b, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_solve() {
        // Identity matrix
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![2.0, 3.0];
        let x = solve(&a.view(), &b.view(), None).unwrap();
        assert_relative_eq!(x[0], 2.0);
        assert_relative_eq!(x[1], 3.0);

        // General 2x2 matrix
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![5.0, 11.0];
        let x = solve(&a.view(), &b.view(), None).unwrap();
        assert_relative_eq!(x[0], 1.0);
        assert_relative_eq!(x[1], 2.0);
    }

    #[test]
    fn test_solve_triangular_lower() {
        // Lower triangular system
        let a = array![[1.0, 0.0], [2.0, 3.0]];
        let b = array![2.0, 8.0];
        let x = solve_triangular(&a.view(), &b.view(), true, false).unwrap();
        assert_relative_eq!(x[0], 2.0);
        assert_relative_eq!(x[1], 4.0 / 3.0);

        // With unit diagonal
        let a = array![[1.0, 0.0], [2.0, 1.0]];
        let b = array![2.0, 6.0];
        let x = solve_triangular(&a.view(), &b.view(), true, true).unwrap();
        assert_relative_eq!(x[0], 2.0);
        assert_relative_eq!(x[1], 2.0);
    }

    #[test]
    fn test_solve_triangular_upper() {
        // Upper triangular system
        let a = array![[3.0, 2.0], [0.0, 1.0]];
        let b = array![8.0, 2.0];
        let x = solve_triangular(&a.view(), &b.view(), false, false).unwrap();
        assert_relative_eq!(x[0], 4.0 / 3.0);
        assert_relative_eq!(x[1], 2.0);

        // With unit diagonal
        let a = array![[1.0, 2.0], [0.0, 1.0]];
        let b = array![6.0, 2.0];
        let x = solve_triangular(&a.view(), &b.view(), false, true).unwrap();
        assert_relative_eq!(x[0], 2.0);
        assert_relative_eq!(x[1], 2.0);
    }
}
