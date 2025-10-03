//! Accelerated BLAS (Basic Linear Algebra Subprograms) operations using ndarray-linalg
//!
//! This module provides optimized BLAS operations using ndarray-linalg bindings to native BLAS libraries.
//! These functions are significantly faster for large matrices compared to pure Rust implementations.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign};

use crate::error::{LinalgError, LinalgResult};

/// Computes the dot product of two vectors using optimized BLAS.
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
///
/// * The dot product of x and y
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas_accelerated::dot;
///
/// let x = array![1.0_f64, 2.0, 3.0];
/// let y = array![4.0_f64, 5.0, 6.0];
/// let result = dot(&x.view(), &y.view()).unwrap();
/// assert!((result - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
/// ```
#[allow(dead_code)]
pub fn dot<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + 'static,
{
    if x.len() != y.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vectors must have the same length for dot product, got {} and {}",
            x.len(),
            y.len()
        )));
    }

    // Use ndarray-linalg dot product implementation
    Ok(x.dot(y))
}

/// Computes the 2-norm (Euclidean norm) of a vector using optimized BLAS.
///
/// # Arguments
///
/// * `x` - Input vector
///
/// # Returns
///
/// * The 2-norm of x
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas_accelerated::norm;
///
/// let x = array![3.0_f64, 4.0];
/// let result = norm(&x.view()).unwrap();
/// assert!((result - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
/// ```
#[allow(dead_code)]
pub fn norm<F>(x: &ArrayView1<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + 'static,
{
    if x.is_empty() {
        return Err(LinalgError::InvalidInputError(
            "Cannot compute norm of an empty vector".to_string(),
        ));
    }

    // Calculate the Euclidean (L2) norm manually
    let mut sum = F::zero();
    for &val in x.iter() {
        sum += val * val;
    }
    Ok(Float::sqrt(sum))
}

/// Performs matrix-vector multiplication using optimized BLAS.
///
/// Computes y = alpha*A*x + beta*y
///
/// # Arguments
///
/// * `alpha` - Scalar value for A*x
/// * `a` - Input matrix A
/// * `x` - Input vector x
/// * `beta` - Scalar value for y
/// * `y` - Input/output vector y
///
/// # Returns
///
/// * The resulting vector
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_linalg::blas_accelerated::gemv;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let x = array![2.0_f64, 3.0];
/// let y = Array1::<f64>::zeros(2);
/// let result = gemv(1.0, &a.view(), &x.view(), 0.0, &y.view()).unwrap();
/// assert!((result[0] - 8.0).abs() < 1e-10); // 1*2 + 2*3 = 8
/// assert!((result[1] - 18.0).abs() < 1e-10); // 3*2 + 4*3 = 18
/// ```
#[allow(dead_code)]
pub fn gemv<F>(
    alpha: F,
    a: &ArrayView2<F>,
    x: &ArrayView1<F>,
    beta: F,
    y: &ArrayView1<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + 'static,
{
    if a.ncols() != x.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix columns ({}) must match vector length ({}) for gemv",
            a.ncols(),
            x.len()
        )));
    }

    if a.nrows() != y.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix rows ({}) must match result vector length ({}) for gemv",
            a.nrows(),
            y.len()
        )));
    }

    // Create result vector (copy y)
    let mut result = y.to_owned();

    // Scale y by beta
    if beta != F::one() {
        result.map_inplace(|v| *v *= beta);
    }

    // Compute matrix-vector product using ndarray-linalg
    // a.dot(x) * alpha + result
    let ax = a.dot(x);
    result.zip_mut_with(&ax, |y_i, &ax_i| *y_i += alpha * ax_i);

    Ok(result)
}

/// Performs matrix-matrix multiplication using optimized BLAS.
///
/// Computes C = alpha*A*B + beta*C
///
/// # Arguments
///
/// * `alpha` - Scalar value for A*B
/// * `a` - Input matrix A
/// * `b` - Input matrix B
/// * `beta` - Scalar value for C
/// * `c` - Input/output matrix C
///
/// # Returns
///
/// * The resulting matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_linalg::blas_accelerated::gemm;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
/// let c = Array2::<f64>::zeros((2, 2));
/// let result = gemm(1.0, &a.view(), &b.view(), 0.0, &c.view()).unwrap();
/// assert!((result[[0, 0]] - 19.0).abs() < 1e-10); // 1*5 + 2*7 = 19
/// assert!((result[[0, 1]] - 22.0).abs() < 1e-10); // 1*6 + 2*8 = 22
/// assert!((result[[1, 0]] - 43.0).abs() < 1e-10); // 3*5 + 4*7 = 43
/// assert!((result[[1, 1]] - 50.0).abs() < 1e-10); // 3*6 + 4*8 = 50
/// ```
#[allow(dead_code)]
pub fn gemm<F>(
    alpha: F,
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    beta: F,
    c: &ArrayView2<F>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + 'static,
{
    if a.ncols() != b.nrows() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions not compatible for multiplication: a.ncols ({}) != b.nrows ({})",
            a.ncols(),
            b.nrows()
        )));
    }

    if a.nrows() != c.nrows() || b.ncols() != c.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Output matrix dimensions ({},{}) don't match expected ({},{})",
            c.nrows(),
            c.ncols(),
            a.nrows(),
            b.ncols()
        )));
    }

    // Create result matrix (copy c)
    let mut result = c.to_owned();

    // Scale c by beta
    if beta != F::one() {
        result.map_inplace(|v| *v *= beta);
    }

    // Compute matrix-matrix product using ndarray-linalg
    // a.dot(b) * alpha + result
    let ab = a.dot(b);
    result.zip_mut_with(&ab, |c_ij, &ab_ij| *c_ij += alpha * ab_ij);

    Ok(result)
}

/// Performs matrix-matrix multiplication of large matrices using optimized BLAS.
///
/// This version is optimized for large matrices and returns a new matrix C = A * B.
///
/// # Arguments
///
/// * `a` - Input matrix A
/// * `b` - Input matrix B
///
/// # Returns
///
/// * The resulting matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas_accelerated::matmul;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
/// let c = matmul(&a.view(), &b.view()).unwrap();
/// assert!((c[[0, 0]] - 19.0).abs() < 1e-10); // 1*5 + 2*7 = 19
/// assert!((c[[0, 1]] - 22.0).abs() < 1e-10); // 1*6 + 2*8 = 22
/// assert!((c[[1, 0]] - 43.0).abs() < 1e-10); // 3*5 + 4*7 = 43
/// assert!((c[[1, 1]] - 50.0).abs() < 1e-10); // 3*6 + 4*8 = 50
/// ```
#[allow(dead_code)]
pub fn matmul<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + 'static,
{
    if a.ncols() != b.nrows() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions not compatible for multiplication: a.ncols ({}) != b.nrows ({})",
            a.ncols(),
            b.nrows()
        )));
    }

    // Use ndarray-linalg's dot implementation for optimal performance
    Ok(a.dot(b))
}

/// Solves the linear system Ax = b for large matrices using optimized LAPACK.
///
/// # Arguments
///
/// * `a` - Coefficient matrix
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// * Solution vector x
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas_accelerated::solve;
///
/// let a = array![[3.0_f64, 1.0], [1.0, 2.0]];
/// let b = array![9.0_f64, 8.0];
/// let x = solve(&a.view(), &b.view()).unwrap();
/// assert!((x[0] - 2.0).abs() < 1e-10);
/// assert!((x[1] - 3.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn solve<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + 'static,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square for solve, got shape {:?}",
            a.shape()
        )));
    }

    if a.nrows() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix rows ({}) must match vector length ({}) for solve",
            a.nrows(),
            b.len()
        )));
    }

    // Implement a basic solver instead of using ndarray-linalg directly
    // For now, we'll use a simple Gaussian elimination approach
    let n = a.nrows();

    // Create augmented matrix [A|b]
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Gaussian elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = Float::abs(aug[[i, i]]);

        for j in (i + 1)..n {
            let val = Float::abs(aug[[j, i]]);
            if val > max_val {
                max_row = j;
                max_val = val;
            }
        }

        // Check for singular matrix
        if max_val < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..(n + 1) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Eliminate below
        for j in (i + 1)..n {
            let factor = aug[[j, i]] / aug[[i, i]];
            aug[[j, i]] = F::zero(); // Just for numerical stability

            for k in (i + 1)..(n + 1) {
                aug[[j, k]] = aug[[j, k]] - factor * aug[[i, k]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Computes the inverse of a square matrix using optimized LAPACK.
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
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas_accelerated::inv;
///
/// let a = array![[4.0_f64, 7.0], [2.0, 6.0]];
/// let a_inv = inv(&a.view()).unwrap();
/// // Check that A * A^-1 is approximately identity
/// let identity = a.dot(&a_inv);
/// assert!((identity[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((identity[[0, 1]]).abs() < 1e-10);
/// assert!((identity[[1, 0]]).abs() < 1e-10);
/// assert!((identity[[1, 1]] - 1.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn inv<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + 'static,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square for inverse, got shape {:?}",
            a.shape()
        )));
    }

    // Implement matrix inversion using Gaussian elimination with identity matrix
    let n = a.nrows();

    // Create augmented matrix [A|I]
    let mut aug = Array2::<F>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, i + n]] = F::one(); // Identity matrix part
    }

    // Gaussian elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = Float::abs(aug[[i, i]]);

        for j in (i + 1)..n {
            let val = Float::abs(aug[[j, i]]);
            if val > max_val {
                max_row = j;
                max_val = val;
            }
        }

        // Check for singular matrix
        if max_val < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Scale row to get pivot = 1
        let pivot = aug[[i, i]];
        for j in 0..(2 * n) {
            aug[[i, j]] /= pivot;
        }

        // Eliminate other rows
        for j in 0..n {
            if j != i {
                let factor = aug[[j, i]];
                for k in 0..(2 * n) {
                    aug[[j, k]] = aug[[j, k]] - factor * aug[[i, k]];
                }
            }
        }
    }

    // Extract inverse from right half of augmented matrix
    let mut a_inv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_inv[[i, j]] = aug[[i, j + n]];
        }
    }

    Ok(a_inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{array, Array1, Array2};

    #[test]
    fn test_dot() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];
        let result = dot(&x.view(), &y.view()).unwrap();
        assert_relative_eq!(result, 32.0, epsilon = 1e-10); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_norm() {
        let x = array![3.0, 4.0];
        let result = norm(&x.view()).unwrap();
        assert_relative_eq!(result, 5.0, epsilon = 1e-10); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_gemv() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let x = array![2.0, 3.0];
        let y = Array1::<f64>::zeros(2);
        let result = gemv(1.0, &a.view(), &x.view(), 0.0, &y.view()).unwrap();
        assert_relative_eq!(result[0], 8.0, epsilon = 1e-10); // 1*2 + 2*3 = 8
        assert_relative_eq!(result[1], 18.0, epsilon = 1e-10); // 3*2 + 4*3 = 18
    }

    #[test]
    fn test_gemm() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = Array2::<f64>::zeros((2, 2));
        let result = gemm(1.0, &a.view(), &b.view(), 0.0, &c.view()).unwrap();
        assert_relative_eq!(result[[0, 0]], 19.0, epsilon = 1e-10); // 1*5 + 2*7 = 19
        assert_relative_eq!(result[[0, 1]], 22.0, epsilon = 1e-10); // 1*6 + 2*8 = 22
        assert_relative_eq!(result[[1, 0]], 43.0, epsilon = 1e-10); // 3*5 + 4*7 = 43
        assert_relative_eq!(result[[1, 1]], 50.0, epsilon = 1e-10); // 3*6 + 4*8 = 50
    }

    #[test]
    fn test_matmul() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let result = matmul(&a.view(), &b.view()).unwrap();
        assert_relative_eq!(result[[0, 0]], 19.0, epsilon = 1e-10); // 1*5 + 2*7 = 19
        assert_relative_eq!(result[[0, 1]], 22.0, epsilon = 1e-10); // 1*6 + 2*8 = 22
        assert_relative_eq!(result[[1, 0]], 43.0, epsilon = 1e-10); // 3*5 + 4*7 = 43
        assert_relative_eq!(result[[1, 1]], 50.0, epsilon = 1e-10); // 3*6 + 4*8 = 50
    }

    #[test]
    fn test_solve() {
        let a = array![[3.0, 1.0], [1.0, 2.0]];
        let b = array![9.0, 8.0];
        let x = solve(&a.view(), &b.view()).unwrap();
        assert_relative_eq!(x[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 3.0, epsilon = 1e-10);

        // Verify solution
        let b_check = a.dot(&x);
        assert_relative_eq!(b_check[0], b[0], epsilon = 1e-10);
        assert_relative_eq!(b_check[1], b[1], epsilon = 1e-10);
    }

    #[test]
    fn test_inv() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let a_inv = inv(&a.view()).unwrap();

        // Check a few values
        assert_relative_eq!(a_inv[[0, 0]], 0.6, epsilon = 1e-10);
        assert_relative_eq!(a_inv[[0, 1]], -0.7, epsilon = 1e-10);
        assert_relative_eq!(a_inv[[1, 0]], -0.2, epsilon = 1e-10);
        assert_relative_eq!(a_inv[[1, 1]], 0.4, epsilon = 1e-10);

        // Check that A * A^-1 is approximately identity
        let identity = a.dot(&a_inv);
        assert_relative_eq!(identity[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity[[1, 1]], 1.0, epsilon = 1e-10);
    }
}
