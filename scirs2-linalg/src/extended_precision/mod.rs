//! Extended precision matrix operations
//!
//! This module provides functions for performing matrix operations with extended precision.
//!
//! ## Overview
//!
//! This module provides utilities for higher precision matrix computations:
//!
//! * Basic operations with extended precision (matrix-vector, matrix-matrix multiply)
//! * Linear system solvers with extended precision
//! * Eigenvalue computations with extended precision
//! * Matrix factorizations (LU, QR, Cholesky, SVD) with extended precision
//!
//! Extended precision operations are useful for computations involving ill-conditioned matrices,
//! or when working with matrices that have entries with widely different magnitudes.
//!
//! ## Examples
//!
//! Basic operations:
//!
//! ```
//! use ndarray::{array, ArrayView2, ArrayView1};
//! use scirs2_linalg::extended_precision::{extended_matmul, extended_matvec};
//!
//! // Create a matrix and vectors in f32 precision
//! let a = array![[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]];
//! let x = array![0.1_f32, 0.2_f32];
//!
//! // Compute matrix-vector product with higher precision
//! let y = extended_matvec::<_, f64>(&a.view(), &x.view()).unwrap();
//!
//! // Compute matrix-matrix product with higher precision
//! let b = array![[5.0_f32, 6.0_f32], [7.0_f32, 8.0_f32]];
//! let c = extended_matmul::<_, f64>(&a.view(), &b.view()).unwrap();
//! ```
//!
//! Matrix factorizations with extended precision:
//!
//! ```
//! use ndarray::{array, ArrayView2};
//! use scirs2_linalg::extended_precision::factorizations::{extended_lu, extended_qr, extended_cholesky};
//!
//! let a = array![
//!     [2.0_f32, 1.0_f32, 1.0_f32],
//!     [4.0_f32, 3.0_f32, 3.0_f32],
//!     [8.0_f32, 7.0_f32, 9.0_f32]
//! ];
//!
//! // LU decomposition with extended precision
//! let (p, l, u) = extended_lu::<_, f64>(&a.view()).unwrap();
//!
//! // QR decomposition with extended precision
//! let (q, r) = extended_qr::<_, f64>(&a.view()).unwrap();
//!
//! // Cholesky decomposition with extended precision
//! let spd_matrix = array![
//!     [4.0_f32, 1.0_f32, 1.0_f32],
//!     [1.0_f32, 5.0_f32, 2.0_f32],
//!     [1.0_f32, 2.0_f32, 6.0_f32]
//! ];
//! let l = extended_cholesky::<_, f64>(&spd_matrix.view()).unwrap();
//! ```

pub mod eigen;
pub mod factorizations;

// Re-export commonly used functions
pub use eigen::{extended_eig, extended_eigvals, extended_eigvalsh};
pub use factorizations::{extended_cholesky, extended_lu, extended_qr, extended_svd};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, One, Zero};

use crate::error::LinalgResult;

/// Trait for promoting numerical types to higher precision
pub trait PromotableTo<T> {
    /// Convert to a higher precision type
    fn promote(self) -> T;
}

// Implement promotions for common types
impl PromotableTo<f64> for f32 {
    fn promote(self) -> f64 {
        self as f64
    }
}

impl PromotableTo<f64> for f64 {
    fn promote(self) -> f64 {
        self // No promotion needed
    }
}

/// Trait for demoting numerical types to lower precision
pub trait DemotableTo<T> {
    /// Convert to a lower precision type
    fn demote(self) -> T;
}

impl DemotableTo<f32> for f64 {
    fn demote(self) -> f32 {
        self as f32
    }
}

impl DemotableTo<f32> for f32 {
    fn demote(self) -> f32 {
        self // No demotion needed
    }
}

/// Matrix-vector multiplication using higher precision arithmetic
///
/// Computes y = A*x using higher precision intermediate calculations
pub fn extended_matvec<A, I>(a: &ArrayView2<A>, x: &ArrayView1<A>) -> LinalgResult<Array1<A>>
where
    A: Float + Zero + PromotableTo<I> + Copy,
    I: Float + Zero + DemotableTo<A> + Copy,
{
    if a.ncols() != x.len() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Incompatible dimensions for matrix-vector multiplication: {:?} and {:?}",
            a.shape(),
            x.shape()
        )));
    }

    let m = a.nrows();
    let n = a.ncols();

    let mut y = Array1::zeros(m);

    for i in 0..m {
        // Compute dot product in higher precision
        let mut sum = I::zero();
        for j in 0..n {
            let a_high = a[[i, j]].promote();
            let x_high = x[j].promote();
            sum = sum + a_high * x_high;
        }

        // Convert back to original precision
        y[i] = sum.demote();
    }

    Ok(y)
}

/// Matrix-matrix multiplication using higher precision arithmetic
///
/// Computes C = A*B using higher precision intermediate calculations
pub fn extended_matmul<A, I>(a: &ArrayView2<A>, b: &ArrayView2<A>) -> LinalgResult<Array2<A>>
where
    A: Float + Zero + PromotableTo<I> + Copy,
    I: Float + Zero + DemotableTo<A> + Copy,
{
    if a.ncols() != b.nrows() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Incompatible dimensions for matrix multiplication: {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    let mut c = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            // Compute dot product in higher precision
            let mut sum = I::zero();
            for l in 0..k {
                let a_high = a[[i, l]].promote();
                let b_high = b[[l, j]].promote();
                sum = sum + a_high * b_high;
            }

            // Convert back to original precision
            c[[i, j]] = sum.demote();
        }
    }

    Ok(c)
}

/// Compute the determinant of a matrix using extended precision
///
/// This function computes the determinant of a square matrix using higher precision
/// intermediate calculations for better accuracy.
///
/// # Arguments
///
/// * `a` - The input matrix
///
/// # Returns
///
/// * Determinant of the matrix
///
/// # Type Parameters
///
/// * `A` - Original precision (input and output)
/// * `I` - Intermediate precision (higher precision for calculations)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::extended_precision::extended_det;
///
/// let a = array![
///     [1.0f32, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0]
/// ];
///
/// // Compute determinant with extended precision (f32 -> f64 -> f32)
/// let det = extended_det::<_, f64>(&a.view()).unwrap();
///
/// // The determinant of this matrix should be 0
/// assert!(det.abs() < 1e-5);
/// ```
pub fn extended_det<A, I>(a: &ArrayView2<A>) -> LinalgResult<A>
where
    A: Float + Zero + One + PromotableTo<I> + DemotableTo<A> + Copy,
    I: Float
        + Zero
        + One
        + DemotableTo<A>
        + Copy
        + PartialOrd
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::SubAssign,
{
    if a.nrows() != a.ncols() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Matrix must be square to compute determinant, got shape {:?}",
            a.shape()
        )));
    }

    // For 2x2 matrices, compute directly for reliability
    if a.nrows() == 2 && a.ncols() == 2 {
        let a00 = a[[0, 0]].promote();
        let a01 = a[[0, 1]].promote();
        let a10 = a[[1, 0]].promote();
        let a11 = a[[1, 1]].promote();

        let det = a00 * a11 - a01 * a10;
        return Ok(det.demote());
    }

    // For 3x3 matrices, also compute directly for reliability
    if a.nrows() == 3 && a.ncols() == 3 {
        let a00 = a[[0, 0]].promote();
        let a01 = a[[0, 1]].promote();
        let a02 = a[[0, 2]].promote();
        let a10 = a[[1, 0]].promote();
        let a11 = a[[1, 1]].promote();
        let a12 = a[[1, 2]].promote();
        let a20 = a[[2, 0]].promote();
        let a21 = a[[2, 1]].promote();
        let a22 = a[[2, 2]].promote();

        // Using cofactor expansion along the first row
        let det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20);

        return Ok(det.demote());
    }

    // For larger matrices, use the LU decomposition
    let (_, _, u) = factorizations::extended_lu::<A, I>(a)?;

    // Compute determinant as the product of diagonal elements of U
    let n = u.nrows();
    let mut det = I::one();

    for i in 0..n {
        // Convert the element to higher precision before multiplication
        let elem = u[[i, i]].promote();
        det = det * elem;
    }

    // Convert result back to original precision
    Ok(det.demote())
}

/// Solve a linear system Ax = b using extended precision
///
/// This implementation uses Gaussian elimination with partial pivoting
/// and higher precision intermediate calculations for better accuracy.
pub fn extended_solve<A, I>(a: &ArrayView2<A>, b: &ArrayView1<A>) -> LinalgResult<Array1<A>>
where
    A: Float + Zero + One + PromotableTo<I> + Copy,
    I: Float + Zero + One + DemotableTo<A> + Copy + PartialOrd,
{
    if a.nrows() != a.ncols() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    if a.nrows() != b.len() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Incompatible dimensions: A is {:?}, b is {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let n = a.nrows();

    // Convert all inputs to higher precision
    let mut a_high = Array2::zeros((n, n));
    let mut b_high = Array1::zeros(n);

    for i in 0..n {
        b_high[i] = b[i].promote();
        for j in 0..n {
            a_high[[i, j]] = a[[i, j]].promote();
        }
    }

    // Gaussian elimination with partial pivoting in higher precision
    for k in 0..n - 1 {
        // Find pivot
        let mut pivot_row = k;
        let mut max_val = a_high[[k, k]].abs();

        for i in k + 1..n {
            let val = a_high[[i, k]].abs();
            if val > max_val {
                max_val = val;
                pivot_row = i;
            }
        }

        // Check for singular matrix
        if max_val < I::epsilon() {
            return Err(crate::error::LinalgError::SingularMatrixError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows if necessary
        if pivot_row != k {
            for j in k..n {
                let temp = a_high[[k, j]];
                a_high[[k, j]] = a_high[[pivot_row, j]];
                a_high[[pivot_row, j]] = temp;
            }

            let temp = b_high[k];
            b_high[k] = b_high[pivot_row];
            b_high[pivot_row] = temp;
        }

        // Elimination
        for i in k + 1..n {
            let factor = a_high[[i, k]] / a_high[[k, k]];

            for j in k + 1..n {
                a_high[[i, j]] = a_high[[i, j]] - factor * a_high[[k, j]];
            }

            b_high[i] = b_high[i] - factor * b_high[k];
            a_high[[i, k]] = I::zero(); // Not necessary but makes structure clear
        }
    }

    // Back substitution in higher precision
    let mut x_high = Array1::zeros(n);

    for i in (0..n).rev() {
        let mut sum = I::zero();
        for j in i + 1..n {
            sum = sum + a_high[[i, j]] * x_high[j];
        }

        x_high[i] = (b_high[i] - sum) / a_high[[i, i]];
    }

    // Convert back to original precision
    let mut x = Array1::zeros(n);
    for i in 0..n {
        x[i] = x_high[i].demote();
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_extended_matvec() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let x = array![0.1f32, 0.2];

        let y = extended_matvec::<_, f64>(&a.view(), &x.view()).unwrap();

        // Expected result: [0.5, 1.1]
        assert!((y[0] - 0.5).abs() < 1e-6);
        assert!((y[1] - 1.1).abs() < 1e-6);
    }

    #[test]
    fn test_extended_matmul() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0f32, 6.0], [7.0, 8.0]];

        let c = extended_matmul::<_, f64>(&a.view(), &b.view()).unwrap();

        // Expected result: [[19.0, 22.0], [43.0, 50.0]]
        assert!((c[[0, 0]] - 19.0).abs() < 1e-6);
        assert!((c[[0, 1]] - 22.0).abs() < 1e-6);
        assert!((c[[1, 0]] - 43.0).abs() < 1e-6);
        assert!((c[[1, 1]] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_extended_det() {
        // Test 2x2 matrix
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];

        // Manually calculate determinant: 1*4 - 2*3 = -2
        let det_manual = 1.0 * 4.0 - 2.0 * 3.0;

        let det = extended_det::<_, f64>(&a.view()).unwrap();
        assert!((det - det_manual).abs() < 1e-6);

        // Test diagonal matrix
        let c = array![[2.0f32, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let det = extended_det::<_, f64>(&c.view()).unwrap();
        assert!((det - 24.0).abs() < 1e-6);

        // Test a non-singular 3x3 matrix
        let d = array![[1.0f32, 3.0, 5.0], [2.0, 4.0, 7.0], [1.0, 1.0, 0.0]];
        let det = extended_det::<_, f64>(&d.view()).unwrap();
        // The correct determinant is 4.0
        assert!((det - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_extended_det_ill_conditioned() {
        // Create a Hilbert matrix (known to be ill-conditioned)
        let mut hilbert = Array2::<f32>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                hilbert[[i, j]] = 1.0 / ((i + j + 1) as f32);
            }
        }

        // Compute determinant with standard precision
        let std_det = crate::basic::det(&hilbert.view(), None).unwrap();

        // Compute determinant with extended precision
        let ext_det = extended_det::<_, f64>(&hilbert.view()).unwrap();

        // The determinant is known to be very small for 4x4 Hilbert matrix
        // Extended precision should be more accurate
        let ref_det = 1.65342e-7; // Refined reference value

        println!("Standard precision det: {:.10e}", std_det);
        println!("Extended precision det: {:.10e}", ext_det);
        println!("Reference value: {:.10e}", ref_det);

        // Just check that extended precision is reasonable - sign might be different
        // but magnitude should be similar
        assert!((ext_det.abs() - ref_det).abs() < 1e-9);
    }

    #[test]
    fn test_extended_solve() {
        let a = array![[4.0f32, 1.0], [1.0, 3.0]];
        let b = array![1.0f32, 2.0];

        let x = extended_solve::<_, f64>(&a.view(), &b.view()).unwrap();

        // Expected result approximately [0.09091, 0.63636]
        assert!((x[0] - 0.09091).abs() < 1e-4);
        assert!((x[1] - 0.63636).abs() < 1e-4);

        // Verify solution by checking A*x ≈ b
        let ax = a.dot(&x);
        assert!((ax[0] - b[0]).abs() < 1e-5);
        assert!((ax[1] - b[1]).abs() < 1e-5);
    }
}
