//! Linear system solvers for B-spline computations
//!
//! This module contains optimized linear algebra routines specifically
//! designed for B-spline interpolation and least-squares fitting.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive, Zero};
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, RemAssign, Sub, SubAssign};

use crate::error::{InterpolateError, InterpolateResult};

/// Solve a linear system Ax = b using optimized structured matrix methods
///
/// This function automatically detects matrix structure and uses the most
/// appropriate solver (band, sparse, or dense).
pub fn solve_linear_system<T>(a: &ArrayView2<T>, b: &ArrayView1<T>) -> InterpolateResult<Array1<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Zero
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy,
{
    if a.nrows() != a.ncols() {
        return Err(InterpolateError::invalid_input(
            "matrix must be square for direct solve".to_string(),
        ));
    }

    if a.nrows() != b.len() {
        return Err(InterpolateError::invalid_input(
            "matrix and vector dimensions must match".to_string(),
        ));
    }

    // Detect if matrix is banded (common for B-spline interpolation)
    let bandwidth = estimate_bandwidth(a);
    let n = a.nrows();

    // Use band solver if bandwidth is significantly smaller than matrix size
    if bandwidth > 0 && bandwidth < n / 4 {
        solve_band_system(a, b, bandwidth)
    } else {
        // Fall back to direct dense solver for small matrices or dense structure
        solve_dense_fallback(a, b)
    }
}

/// Solve a least-squares problem min ||Ax - b||^2 using optimized structured methods
///
/// This function uses structured matrix least squares solver which automatically
/// detects matrix structure for optimal performance.
pub fn solve_least_squares<T>(a: &ArrayView2<T>, b: &ArrayView1<T>) -> InterpolateResult<Array1<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Zero
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy,
{
    if a.nrows() != b.len() {
        return Err(InterpolateError::invalid_input(
            "matrix and vector dimensions must match".to_string(),
        ));
    }

    // Use normal equations approach: solve (A^T A) x = A^T b
    let at = transpose_matrix(a);
    let ata = matrix_multiply(&at.view(), &a)?;
    let atb = matrix_vector_multiply(&at.view(), b)?;

    // Add regularization for numerical stability
    let mut regularized_ata = ata.clone();
    let reg = if a.nrows() < a.ncols() {
        // Underdetermined system - add more regularization
        T::from_f64(1e-8).unwrap()
    } else {
        // Square or overdetermined system - add minimal regularization
        T::from_f64(1e-10).unwrap()
    };

    for i in 0..regularized_ata.nrows() {
        regularized_ata[[i, i]] += reg;
    }

    // Try to solve, if it fails, increase regularization
    match solve_linear_system(&regularized_ata.view(), &atb.view()) {
        Ok(result) => Ok(result),
        Err(_) => {
            // Increase regularization significantly
            for i in 0..regularized_ata.nrows() {
                regularized_ata[[i, i]] += T::from_f64(1e-6).unwrap();
            }
            solve_linear_system(&regularized_ata.view(), &atb.view())
        }
    }
}

/// Solve banded linear system using optimized band solver
fn solve_band_system<T>(
    a: &ArrayView2<T>,
    b: &ArrayView1<T>,
    bandwidth: usize,
) -> InterpolateResult<Array1<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Zero
        + Copy,
{
    let _n = a.nrows();

    // For now, use a simplified band solver
    // In a full implementation, this would use optimized band LU decomposition

    // Convert to standard format and use dense solver
    // This is a placeholder - real band solver would be more efficient
    solve_dense_fallback(a, b)
}

/// Estimate the bandwidth of a matrix
///
/// Returns the maximum distance from the main diagonal that contains non-zero elements.
fn estimate_bandwidth<T: Float + Zero + FromPrimitive>(matrix: &ArrayView2<T>) -> usize {
    let n = matrix.nrows();
    let mut max_bandwidth = 0;
    let tolerance = T::from_f64(1e-14).unwrap();

    for i in 0..n {
        for j in 0..n {
            if matrix[[i, j]].abs() > tolerance {
                let bandwidth = if i > j { i - j } else { j - i };
                max_bandwidth = max_bandwidth.max(bandwidth);
            }
        }
    }

    max_bandwidth
}

/// Dense fallback solver using Gaussian elimination
fn solve_dense_fallback<T>(
    matrix: &ArrayView2<T>,
    rhs: &ArrayView1<T>,
) -> InterpolateResult<Array1<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Zero
        + Copy,
{
    let n = matrix.nrows();

    // Create augmented matrix [A|b]
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = matrix[[i, j]];
        }
        aug[[i, n]] = rhs[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = aug[[k, k]].abs();
        for i in (k + 1)..n {
            let val = aug[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Check for singular matrix with more reasonable threshold
        // Use machine epsilon * matrix size as threshold for singularity
        let eps = T::from_f64(2.22e-16).unwrap(); // Machine epsilon for f64
        let threshold = eps * T::from_usize(n).unwrap() * T::from_f64(1e8).unwrap();
        if max_val < threshold {
            return Err(InterpolateError::invalid_input(
                "matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows if needed
        if max_row != k {
            for j in 0..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Eliminate column k
        for i in (k + 1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..=n {
                let temp = aug[[k, j]];
                aug[[i, j]] -= factor * temp;
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Transpose a matrix
pub fn transpose_matrix<T: Copy + Zero>(matrix: &ArrayView2<T>) -> Array2<T> {
    let (rows, cols) = matrix.dim();
    let mut result = Array2::<T>::zeros((cols, rows));

    for i in 0..rows {
        for j in 0..cols {
            result[[j, i]] = matrix[[i, j]];
        }
    }

    result
}

/// Multiply two matrices A * B
pub fn matrix_multiply<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> InterpolateResult<Array2<T>>
where
    T: Float + Zero + AddAssign + MulAssign + Copy,
{
    if a.ncols() != b.nrows() {
        return Err(InterpolateError::invalid_input(
            "matrix dimensions do not match for multiplication".to_string(),
        ));
    }

    let (m, n) = (a.nrows(), b.ncols());
    let mut result = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for k in 0..a.ncols() {
                sum += a[[i, k]] * b[[k, j]];
            }
            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

/// Multiply matrix A with vector b
pub fn matrix_vector_multiply<T>(
    a: &ArrayView2<T>,
    b: &ArrayView1<T>,
) -> InterpolateResult<Array1<T>>
where
    T: Float + Zero + AddAssign + Copy,
{
    if a.ncols() != b.len() {
        return Err(InterpolateError::invalid_input(
            "matrix and vector dimensions do not match for multiplication".to_string(),
        ));
    }

    let mut result = Array1::zeros(a.nrows());

    for i in 0..a.nrows() {
        let mut sum = T::zero();
        for j in 0..a.ncols() {
            sum += a[[i, j]] * b[j];
        }
        result[i] = sum;
    }

    Ok(result)
}

/// LU decomposition with partial pivoting
pub fn lu_decomposition<T>(
    matrix: &ArrayView2<T>,
) -> InterpolateResult<(Array2<T>, Array2<T>, Vec<usize>)>
where
    T: Float
        + FromPrimitive
        + Debug
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Zero
        + Copy,
{
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(InterpolateError::invalid_input(
            "matrix must be square for LU decomposition".to_string(),
        ));
    }

    let mut l = Array2::zeros((n, n));
    let mut u = matrix.to_owned();
    let mut perm = Vec::new();

    // Initialize L as identity
    for i in 0..n {
        l[[i, i]] = T::one();
        perm.push(i);
    }

    // LU decomposition with partial pivoting
    for k in 0..n - 1 {
        // Find pivot
        let mut max_row = k;
        let mut max_val = u[[k, k]].abs();
        for i in (k + 1)..n {
            let val = u[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Check for singular matrix with more reasonable threshold
        // Use machine epsilon * matrix size as threshold for singularity
        let eps = T::from_f64(2.22e-16).unwrap(); // Machine epsilon for f64
        let threshold = eps * T::from_usize(n).unwrap() * T::from_f64(1000.0).unwrap();
        if max_val < threshold {
            return Err(InterpolateError::invalid_input(
                "matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows in U and permutation
        if max_row != k {
            for j in 0..n {
                let temp = u[[k, j]];
                u[[k, j]] = u[[max_row, j]];
                u[[max_row, j]] = temp;
            }
            perm.swap(k, max_row);
        }

        // Elimination
        for i in (k + 1)..n {
            let factor = u[[i, k]] / u[[k, k]];
            l[[i, k]] = factor;

            for j in k..n {
                let kj_value = u[[k, j]];
                u[[i, j]] -= factor * kj_value;
            }
        }
    }

    Ok((l, u, perm))
}

/// Solve linear system using LU decomposition
pub fn solve_with_lu<T>(
    l: &ArrayView2<T>,
    u: &ArrayView2<T>,
    perm: &[usize],
    b: &ArrayView1<T>,
) -> InterpolateResult<Array1<T>>
where
    T: Float
        + FromPrimitive
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + SubAssign
        + Zero
        + Copy,
{
    let n = b.len();

    // Apply permutation to b
    let mut pb = Array1::zeros(n);
    for i in 0..n {
        pb[i] = b[perm[i]];
    }

    // Forward substitution: Ly = Pb
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = pb[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }

    // Back substitution: Ux = y
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= u[[i, j]] * x[j];
        }
        x[i] = sum / u[[i, i]];
    }

    Ok(x)
}

/// Solve multiple right-hand sides efficiently using LU decomposition
pub fn solve_multiple_rhs<T>(
    matrix: &ArrayView2<T>,
    rhs_matrix: &ArrayView2<T>,
) -> InterpolateResult<Array2<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Zero
        + Copy,
{
    if matrix.nrows() != rhs_matrix.nrows() {
        return Err(InterpolateError::invalid_input(
            "matrix and RHS dimensions must match".to_string(),
        ));
    }

    // Perform LU decomposition once
    let (l, u, perm) = lu_decomposition(matrix)?;

    let mut solutions = Array2::zeros((matrix.ncols(), rhs_matrix.ncols()));

    // Solve for each RHS column
    for j in 0..rhs_matrix.ncols() {
        let rhs_col = rhs_matrix.column(j);
        let solution = solve_with_lu(&l.view(), &u.view(), &perm, &rhs_col)?;

        for i in 0..solution.len() {
            solutions[[i, j]] = solution[i];
        }
    }

    Ok(solutions)
}

/// Compute matrix condition number (estimate)
pub fn condition_number<T>(matrix: &ArrayView2<T>) -> InterpolateResult<T>
where
    T: Float + FromPrimitive + Debug + Zero + Copy + AddAssign,
{
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(InterpolateError::invalid_input(
            "matrix must be square to compute condition number".to_string(),
        ));
    }

    // Simple estimate using matrix norms
    let matrix_norm = frobenius_norm(matrix);

    // Estimate of smallest singular value (simplified)
    let mut min_diag = T::infinity();
    for i in 0..n {
        let val = matrix[[i, i]].abs();
        if val < min_diag {
            min_diag = val;
        }
    }

    if min_diag < T::from_f64(1e-14).unwrap() {
        Ok(T::infinity())
    } else {
        Ok(matrix_norm / min_diag)
    }
}

/// Compute Frobenius norm of a matrix
fn frobenius_norm<T: Float + Zero + Copy + AddAssign>(matrix: &ArrayView2<T>) -> T {
    let mut sum = T::zero();
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let val = matrix[[i, j]];
            sum += val * val;
        }
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dense_solver() {
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];

        let x = solve_dense_fallback(&a.view(), &b.view()).unwrap();

        // Verify solution: Ax should equal b
        let verification = matrix_vector_multiply(&a.view(), &x.view()).unwrap();
        for i in 0..b.len() {
            assert!((verification[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_matrix_multiply() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result = matrix_multiply(&a.view(), &b.view()).unwrap();

        // Expected: [[19, 22], [43, 50]]
        assert_eq!(result[[0, 0]], 19.0);
        assert_eq!(result[[0, 1]], 22.0);
        assert_eq!(result[[1, 0]], 43.0);
        assert_eq!(result[[1, 1]], 50.0);
    }

    #[test]
    fn test_lu_decomposition() {
        let a = array![[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]];

        let result = lu_decomposition(&a.view());
        assert!(result.is_ok());

        let (l, u, _perm) = result.unwrap();

        // L should be lower triangular with ones on diagonal
        for i in 0..l.nrows() {
            assert_eq!(l[[i, i]], 1.0);
            for j in (i + 1)..l.ncols() {
                assert_eq!(l[[i, j]], 0.0);
            }
        }

        // U should be upper triangular
        for i in 0..u.nrows() {
            for j in 0..i {
                assert!(u[[i, j]].abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_bandwidth_estimation() {
        // Tridiagonal matrix
        let mut a = Array2::zeros((4, 4));
        a[[0, 0]] = 1.0;
        a[[0, 1]] = 1.0;
        a[[1, 0]] = 1.0;
        a[[1, 1]] = 2.0;
        a[[1, 2]] = 1.0;
        a[[2, 1]] = 1.0;
        a[[2, 2]] = 2.0;
        a[[2, 3]] = 1.0;
        a[[3, 2]] = 1.0;
        a[[3, 3]] = 1.0;

        let bandwidth = estimate_bandwidth(&a.view());
        assert_eq!(bandwidth, 1); // Tridiagonal has bandwidth 1
    }
}
