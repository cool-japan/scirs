//! BLAS (Basic Linear Algebra Subprograms) interface
//!
//! This module provides interfaces to BLAS functions.
//!
//! The BLAS (Basic Linear Algebra Subprograms) are routines that provide standard
//! building blocks for performing basic vector and matrix operations.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign};

/// Computes the dot product of two vectors (Level 1 BLAS operation).
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
/// use scirs2_linalg::blas::dot;
///
/// let x = array![1.0_f64, 2.0, 3.0];
/// let y = array![4.0_f64, 5.0, 6.0];
/// let result = dot(&x.view(), &y.view());
/// assert!((result - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
/// ```
#[allow(dead_code)]
pub fn dot<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> F
where
    F: Float + NumAssign,
{
    if x.len() != y.len() {
        panic!("Vectors must have the same length for dot product");
    }

    let mut result = F::zero();
    for i in 0..x.len() {
        result += x[i] * y[i];
    }
    result
}

/// Computes the 2-norm (Euclidean norm) of a vector (Level 1 BLAS operation).
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
/// use scirs2_linalg::blas::nrm2;
///
/// let x = array![3.0_f64, 4.0];
/// let result = nrm2(&x.view());
/// assert!((result - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
/// ```
#[allow(dead_code)]
pub fn nrm2<F>(x: &ArrayView1<F>) -> F
where
    F: Float + NumAssign,
{
    let mut result = F::zero();
    for i in 0..x.len() {
        result += x[i] * x[i];
    }
    result.sqrt()
}

/// Computes the sum of absolute values of a vector (Level 1 BLAS operation).
///
/// # Arguments
///
/// * `x` - Input vector
///
/// # Returns
///
/// * The sum of absolute values of x
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas::asum;
///
/// let x = array![1.0_f64, -2.0, 3.0];
/// let result = asum(&x.view());
/// assert!((result - 6.0).abs() < 1e-10); // |1| + |-2| + |3| = 6
/// ```
#[allow(dead_code)]
pub fn asum<F>(x: &ArrayView1<F>) -> F
where
    F: Float + NumAssign,
{
    let mut result = F::zero();
    for i in 0..x.len() {
        result += x[i].abs();
    }
    result
}

/// Finds the index of the element with the largest absolute value (Level 1 BLAS operation).
///
/// # Arguments
///
/// * `x` - Input vector
///
/// # Returns
///
/// * The index of the element with the largest absolute value (0-based)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas::iamax;
///
/// let x = array![1.0_f64, -5.0, 3.0];
/// let result = iamax(&x.view());
/// assert_eq!(result, 1); // -5.0 has the largest absolute value
/// ```
#[allow(dead_code)]
pub fn iamax<F>(x: &ArrayView1<F>) -> usize
where
    F: Float + NumAssign,
{
    if x.is_empty() {
        panic!("Cannot find maximum of an empty vector");
    }

    let mut max_idx = 0;
    let mut max_val = x[0].abs();

    for i in 1..x.len() {
        let abs_val = x[i].abs();
        if abs_val > max_val {
            max_val = abs_val;
            max_idx = i;
        }
    }
    max_idx
}

/// Performs a scaled vector addition (Level 1 BLAS operation): y = alpha*x + y
///
/// # Arguments
///
/// * `alpha` - Scalar value
/// * `x` - Input vector to be scaled
/// * `y` - Input/output vector
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas::axpy;
///
/// let x = array![1.0_f64, 2.0, 3.0];
/// let mut y = array![4.0_f64, 5.0, 6.0];
/// axpy(2.0, &x.view(), &mut y);
/// assert!((y[0] - 6.0).abs() < 1e-10); // 2*1 + 4 = 6
/// assert!((y[1] - 9.0).abs() < 1e-10); // 2*2 + 5 = 9
/// assert!((y[2] - 12.0).abs() < 1e-10); // 2*3 + 6 = 12
/// ```
#[allow(dead_code)]
pub fn axpy<F>(alpha: F, x: &ArrayView1<F>, y: &mut Array1<F>)
where
    F: Float + NumAssign,
{
    if x.len() != y.len() {
        panic!("Vectors must have the same length for axpy operation");
    }

    for i in 0..x.len() {
        y[i] += alpha * x[i];
    }
}

/// Performs matrix-vector multiplication (Level 2 BLAS operation): y = alpha*A*x + beta*y
///
/// # Arguments
///
/// * `alpha` - Scalar value for A*x
/// * `a` - Input matrix A
/// * `x` - Input vector x
/// * `beta` - Scalar value for y
/// * `y` - Input/output vector y
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_linalg::blas::gemv;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let x = array![2.0_f64, 3.0];
/// let mut y = Array1::<f64>::zeros(2);
/// gemv(1.0, &a.view(), &x.view(), 0.0, &mut y);
/// assert!((y[0] - 8.0).abs() < 1e-10); // 1*2 + 2*3 = 8
/// assert!((y[1] - 18.0).abs() < 1e-10); // 3*2 + 4*3 = 18
/// ```
#[allow(dead_code)]
pub fn gemv<F>(alpha: F, a: &ArrayView2<F>, x: &ArrayView1<F>, beta: F, y: &mut Array1<F>)
where
    F: Float + NumAssign,
{
    if a.ncols() != x.len() || a.nrows() != y.len() {
        panic!("Incompatible dimensions for matrix-vector multiplication");
    }

    // Scale y by beta
    if beta != F::one() {
        for i in 0..y.len() {
            y[i] *= beta;
        }
    }

    // Compute matrix-vector product
    for i in 0..a.nrows() {
        let row = a.slice(scirs2_core::ndarray::s![i, ..]);
        let mut sum = F::zero();
        for j in 0..x.len() {
            sum += row[j] * x[j];
        }
        y[i] += alpha * sum;
    }
}

/// Performs matrix-matrix multiplication (Level 3 BLAS operation): C = alpha*A*B + beta*C
///
/// # Arguments
///
/// * `alpha` - Scalar value for A*B
/// * `a` - Input matrix A
/// * `b` - Input matrix B
/// * `beta` - Scalar value for C
/// * `c` - Input/output matrix C
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_linalg::blas::gemm;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
/// let mut c = Array2::<f64>::zeros((2, 2));
/// gemm(1.0, &a.view(), &b.view(), 0.0, &mut c);
/// assert!((c[[0, 0]] - 19.0).abs() < 1e-10); // 1*5 + 2*7 = 19
/// assert!((c[[0, 1]] - 22.0).abs() < 1e-10); // 1*6 + 2*8 = 22
/// assert!((c[[1, 0]] - 43.0).abs() < 1e-10); // 3*5 + 4*7 = 43
/// assert!((c[[1, 1]] - 50.0).abs() < 1e-10); // 3*6 + 4*8 = 50
/// ```
#[allow(dead_code)]
pub fn gemm<F>(alpha: F, a: &ArrayView2<F>, b: &ArrayView2<F>, beta: F, c: &mut Array2<F>)
where
    F: Float + NumAssign,
{
    if a.ncols() != b.nrows() || a.nrows() != c.nrows() || b.ncols() != c.ncols() {
        panic!("Incompatible dimensions for matrix-matrix multiplication");
    }

    // Scale C by beta
    if beta != F::one() {
        for i in 0..c.nrows() {
            for j in 0..c.ncols() {
                c[[i, j]] *= beta;
            }
        }
    }

    // Compute matrix-matrix product
    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            let mut sum = F::zero();
            for k in 0..a.ncols() {
                sum += a[[i, k]] * b[[k, j]];
            }
            c[[i, j]] += alpha * sum;
        }
    }
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
        let result = dot(&x.view(), &y.view());
        assert_relative_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_nrm2() {
        let x = array![3.0, 4.0];
        let result = nrm2(&x.view());
        assert_relative_eq!(result, 5.0); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_asum() {
        let x = array![1.0, -2.0, 3.0];
        let result = asum(&x.view());
        assert_relative_eq!(result, 6.0); // |1| + |-2| + |3| = 6
    }

    #[test]
    fn test_iamax() {
        let x = array![1.0, -5.0, 3.0];
        let result = iamax(&x.view());
        assert_eq!(result, 1); // -5.0 has the largest absolute value
    }

    #[test]
    fn test_axpy() {
        let x = array![1.0, 2.0, 3.0];
        let mut y = array![4.0, 5.0, 6.0];
        axpy(2.0, &x.view(), &mut y);
        assert_relative_eq!(y[0], 6.0); // 2*1 + 4 = 6
        assert_relative_eq!(y[1], 9.0); // 2*2 + 5 = 9
        assert_relative_eq!(y[2], 12.0); // 2*3 + 6 = 12
    }

    #[test]
    fn test_gemv() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let x = array![2.0, 3.0];
        let mut y = Array1::zeros(2);
        gemv(1.0, &a.view(), &x.view(), 0.0, &mut y);
        assert_relative_eq!(y[0], 8.0); // 1*2 + 2*3 = 8
        assert_relative_eq!(y[1], 18.0); // 3*2 + 4*3 = 18
    }

    #[test]
    fn test_gemm() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let mut c = Array2::zeros((2, 2));
        gemm(1.0, &a.view(), &b.view(), 0.0, &mut c);
        assert_relative_eq!(c[[0, 0]], 19.0); // 1*5 + 2*7 = 19
        assert_relative_eq!(c[[0, 1]], 22.0); // 1*6 + 2*8 = 22
        assert_relative_eq!(c[[1, 0]], 43.0); // 3*5 + 4*7 = 43
        assert_relative_eq!(c[[1, 1]], 50.0); // 3*6 + 4*8 = 50
    }
}
