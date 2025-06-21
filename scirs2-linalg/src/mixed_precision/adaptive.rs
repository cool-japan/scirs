//! Adaptive mixed-precision algorithms for advanced linear algebra
//!
//! This module provides sophisticated algorithms that adapt their precision
//! strategy based on the problem characteristics, including iterative refinement,
//! condition number estimation, and advanced matrix decompositions.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumAssign, NumCast, One, ToPrimitive, Zero};
use std::fmt::Debug;

use super::conversions::{convert, convert_2d};
use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};

/// Solve a linear system using mixed precision
///
/// Converts to higher precision for computation and returns result in desired precision.
///
/// # Arguments
/// * `a` - Coefficient matrix in precision A
/// * `b` - Right-hand side vector in precision B
///
/// # Returns
/// * Solution vector in precision C
///
/// # Type Parameters
/// * `A` - Input matrix precision
/// * `B` - Input vector precision
/// * `C` - Output vector precision
/// * `H` - High precision used for computation
///
/// # Examples
/// ```
/// use ndarray::{array, ArrayView1, ArrayView2};
/// use scirs2_linalg::mixed_precision::adaptive::mixed_precision_solve;
///
/// // Create a simple system in f32 precision
/// let a_f32 = array![[2.0f32, 1.0f32], [1.0f32, 3.0f32]];
/// let b_f32 = array![5.0f32, 8.0f32];
///
/// // Solve using internal f64 precision
/// let x = mixed_precision_solve::<f32, f32, f32, f64>(
///     &a_f32.view(),
///     &b_f32.view()
/// ).unwrap();
///
/// assert_eq!(x.len(), 2);
/// ```
pub fn mixed_precision_solve<A, B, C, H>(
    a: &ArrayView2<A>,
    b: &ArrayView1<B>,
) -> LinalgResult<Array1<C>>
where
    A: Clone + Debug + ToPrimitive + Copy,
    B: Clone + Debug + ToPrimitive + Copy,
    C: Clone + Zero + NumCast + Debug,
    H: Float + Clone + NumCast + Debug + Zero + ToPrimitive + NumAssign,
{
    // Check dimensions
    let a_shape = a.shape();
    if a_shape[0] != a_shape[1] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square, got shape {:?}",
            a_shape
        )));
    }

    if a_shape[0] != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix rows ({}) must match vector length ({})",
            a_shape[0],
            b.len()
        )));
    }

    // Convert to high precision
    let a_high = convert_2d::<A, H>(a);
    let b_high = convert::<B, H>(b);

    // Perform Gaussian elimination with partial pivoting in high precision
    let n = a_shape[0];
    let mut aug = Array2::<H>::zeros((n, n + 1));

    // Create augmented matrix [A|b]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a_high[[i, j]];
        }
        aug[[i, n]] = b_high[i];
    }

    // Gaussian elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = aug[[i, i]].abs();

        for j in i + 1..n {
            let val = aug[[j, i]].abs();
            if val > max_val {
                max_row = j;
                max_val = val;
            }
        }

        // Check for singular matrix
        if max_val < H::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular".to_string(),
            ));
        }

        // Swap rows if needed
        if max_row != i {
            for j in i..=n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Eliminate
        for j in i + 1..n {
            let factor = aug[[j, i]] / aug[[i, i]];
            aug[[j, i]] = H::zero(); // Set to 0 explicitly to avoid floating-point errors

            for k in i + 1..=n {
                aug[[j, k]] = aug[[j, k]] - factor * aug[[i, k]];
            }
        }
    }

    // Back substitution
    let mut x_high = Array1::<H>::zeros(n);

    for i in (0..n).rev() {
        let mut sum = H::zero();
        for j in i + 1..n {
            sum += aug[[i, j]] * x_high[j];
        }
        x_high[i] = (aug[[i, n]] - sum) / aug[[i, i]];
    }

    // Convert back to desired output precision
    let mut result = Array1::<C>::zeros(n);
    for (i, &val) in x_high.iter().enumerate() {
        result[i] = C::from(val).unwrap_or_else(|| C::zero());
    }

    Ok(result)
}

/// Compute the condition number of a matrix using mixed precision
///
/// The condition number is computed as the ratio of the largest to smallest
/// singular value using SVD with internal higher precision computation.
///
/// # Arguments
/// * `a` - Input matrix in precision A
/// * `p` - Norm to use (None for 2-norm)
///
/// # Returns
/// * Condition number in precision C
///
/// # Type Parameters
/// * `A` - Input matrix precision
/// * `C` - Output scalar precision
/// * `H` - High precision used for computation
///
/// # Examples
/// ```
/// use ndarray::array;
/// use scirs2_linalg::mixed_precision::adaptive::mixed_precision_cond;
///
/// // Create a matrix with poor condition number
/// let a_f32 = array![
///     [1.0f32, 2.0f32],
///     [2.0f32, 4.0001f32]  // Almost linearly dependent rows
/// ];
///
/// // Compute condition number with internal f64 precision
/// let cond = mixed_precision_cond::<f32, f32, f64>(&a_f32.view(), None).unwrap();
///
/// // The condition number should indicate poor conditioning
/// println!("Condition number: {}", cond);
/// ```
pub fn mixed_precision_cond<A, C, H>(a: &ArrayView2<A>, p: Option<H>) -> LinalgResult<C>
where
    A: Clone + Debug + ToPrimitive + Copy,
    C: Clone + Zero + NumCast + Debug,
    H: Float
        + Clone
        + NumCast
        + Debug
        + ToPrimitive
        + 'static
        + std::iter::Sum
        + NumAssign
        + ndarray::ScalarOperand,
{
    // Convert to high precision
    let a_high = convert_2d::<A, H>(a);

    // Compute SVD in high precision
    let (_, s, _) = svd(&a_high.view(), false, None)?;

    // Find the largest and smallest singular values
    let s_max = s.iter().cloned().fold(H::zero(), |a, b| a.max(b));
    let s_min = s
        .iter()
        .cloned()
        .filter(|&x| x > H::epsilon())
        .fold(H::infinity(), |a, b| a.min(b));

    // Calculate condition number (based on norm parameter)
    let cond = match p {
        // 2-norm condition number is the ratio of largest to smallest singular value
        None => s_max / s_min,
        // For other norms, we would need to implement different calculations
        Some(_) => {
            return Err(LinalgError::NotImplementedError(
                "Only 2-norm condition number is currently implemented".to_string(),
            ))
        }
    };

    // Convert back to desired output precision
    C::from(cond).ok_or_else(|| {
        LinalgError::ComputationError(
            "Failed to convert condition number to output type".to_string(),
        )
    })
}

/// Solve a linear system with iterative refinement for improved precision
///
/// This function uses mixed-precision iterative refinement to solve a linear system Ax = b.
/// It performs the following steps:
/// 1. Convert the input matrix and vector to lower precision (working precision)
/// 2. Compute an initial solution in lower precision
/// 3. Calculate the residual in higher precision
/// 4. Refine the solution iteratively until convergence or maximum iterations
///
/// # Arguments
///
/// * `a` - Input matrix of precision A
/// * `b` - Input vector of precision B
/// * `max_iter` - Maximum number of refinement iterations (default: 10)
/// * `tol` - Tolerance for convergence (default: 1e-8 in precision H)
///
/// # Returns
///
/// * Solution vector in precision C
///
/// # Type Parameters
///
/// * `A` - Input matrix precision
/// * `B` - Input vector precision
/// * `C` - Output solution precision
/// * `H` - Higher precision used for refinement
/// * `W` - Working (lower) precision used for initial solution
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::mixed_precision::adaptive::iterative_refinement_solve;
///
/// // Solve a linear system Ax = b
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![5.0_f64, 11.0];
///
/// // Use f32 as working precision and f64 as higher precision
/// let x = iterative_refinement_solve::<f64, f64, f64, f64, f32>(
///     &a.view(), &b.view(), None, None
/// ).unwrap();
///
/// // The exact solution is [1.0, 2.0]
/// assert!((x[0] - 1.0).abs() < 1e-10);
/// assert!((x[1] - 2.0).abs() < 1e-10);
/// ```
pub fn iterative_refinement_solve<A, B, C, H, W>(
    a: &ArrayView2<A>,
    b: &ArrayView1<B>,
    max_iter: Option<usize>,
    tol: Option<H>,
) -> LinalgResult<Array1<C>>
where
    A: Float + NumAssign + Debug + 'static,
    B: Float + NumAssign + Debug + 'static,
    C: Float + NumAssign + Debug + 'static,
    H: Float + NumAssign + Debug + 'static + std::iter::Sum + ndarray::ScalarOperand,
    W: Float + NumAssign + Debug + 'static + std::iter::Sum + One,
    A: NumCast,
    B: NumCast,
    C: NumCast,
    H: NumCast,
    W: NumCast,
{
    // Set default values
    let max_iter = max_iter.unwrap_or(10);
    let tol = tol.unwrap_or(NumCast::from(1e-8).unwrap());

    // Check dimensions
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Expected square matrix for iterative refinement, got {:?}",
            a.shape()
        )));
    }
    if a.nrows() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix shape {:?} is incompatible with vector length {}",
            a.shape(),
            b.len()
        )));
    }

    let n = a.nrows();

    // Convert matrix and vector to higher precision for accurate computation
    let a_h: Array2<H> = convert_2d(a);
    let b_h: Array1<H> = convert(b);

    // Convert matrix and vector to working (lower) precision
    let a_w: Array2<W> = convert_2d(a);
    let b_w: Array1<W> = convert(b);

    // Compute initial solution in working precision
    use crate::solve::solve;
    let x_w: Array1<W> = solve(&a_w.view(), &b_w.view(), None)?;

    // Convert initial solution to higher precision
    let mut x_h: Array1<H> = convert(&x_w.view());

    // Iterative refinement
    for _iter in 0..max_iter {
        // Compute residual r = b - A*x in higher precision
        let ax_h = a_h.dot(&x_h);
        let mut r_h = b_h.clone();

        for i in 0..n {
            r_h[i] -= ax_h[i];
        }

        // Check convergence
        let r_norm = r_h.iter().fold(H::zero(), |max, &val| {
            let abs_val = val.abs();
            if abs_val > max {
                abs_val
            } else {
                max
            }
        });

        if r_norm < tol {
            break;
        }

        // Convert residual to working precision
        let r_w: Array1<W> = convert(&r_h.view());

        // Solve for correction in working precision: A * dx = r
        let dx_w = solve(&a_w.view(), &r_w.view(), None)?;

        // Convert correction to higher precision and apply
        let dx_h: Array1<H> = convert(&dx_w.view());

        for i in 0..n {
            x_h[i] += dx_h[i];
        }
    }

    // Convert solution to output precision
    let x_c: Array1<C> = convert(&x_h.view());

    Ok(x_c)
}

/// Perform mixed-precision QR decomposition
///
/// This function computes the QR decomposition of a matrix using a higher precision
/// for internal computations while accepting inputs and producing outputs in specified
/// precisions. It employs the Householder reflection method for numerical stability.
///
/// # Arguments
///
/// * `a` - Input matrix of precision A
///
/// # Returns
///
/// * Tuple (Q, R) where:
///   - Q is an orthogonal matrix of precision C
///   - R is an upper triangular matrix of precision C
///
/// # Type Parameters
///
/// * `A` - Input matrix precision
/// * `C` - Output matrices precision
/// * `H` - Higher precision used for computation
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::mixed_precision::adaptive::mixed_precision_qr;
///
/// // Create a matrix in f32 precision
/// let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
///
/// // Perform QR decomposition using f64 precision internally
/// let (q, r) = mixed_precision_qr::<_, f32, f64>(&a.view()).unwrap();
///
/// // Verify Q is orthogonal: Q^T * Q ≈ I
/// let qt = q.t();
/// let qtq = qt.dot(&q);
///
/// for i in 0..2 {
///     for j in 0..2 {
///         let expected = if i == j { 1.0 } else { 0.0 };
///         assert!((qtq[[i, j]] - expected).abs() < 1e-5);
///     }
/// }
/// ```
pub fn mixed_precision_qr<A, C, H>(a: &ArrayView2<A>) -> LinalgResult<(Array2<C>, Array2<C>)>
where
    A: Float + NumAssign + Debug + 'static,
    C: Float + NumAssign + Debug + 'static,
    H: Float + NumAssign + Debug + 'static + std::iter::Sum + ndarray::ScalarOperand,
    A: NumCast,
    C: NumCast,
    H: NumCast,
{
    // Convert input matrix to higher precision
    let a_h: Array2<H> = convert_2d(a);

    let m = a_h.nrows();
    let n = a_h.ncols();

    // Initialize higher precision matrices
    let mut q_h = Array2::<H>::eye(m);
    let mut r_h = a_h.clone();

    // Householder QR decomposition algorithm with higher precision
    for k in 0..std::cmp::min(m - 1, n) {
        let mut v = Array1::<H>::zeros(m - k);

        // Compute Householder vector
        let mut norm_x = H::zero();
        for i in k..m {
            norm_x += r_h[[i, k]] * r_h[[i, k]];
        }
        norm_x = norm_x.sqrt();

        // Skip if column is already zeros below diagonal
        if norm_x <= NumCast::from(1e-15).unwrap() {
            continue;
        }

        // Compute Householder vector v
        let sign = if r_h[[k, k]] < H::zero() {
            H::one()
        } else {
            -H::one()
        };
        let norm_x_with_sign = sign * norm_x;

        for i in 0..m - k {
            if i == 0 {
                v[i] = r_h[[k, k]] - norm_x_with_sign;
            } else {
                v[i] = r_h[[k + i, k]];
            }
        }

        // Normalize v
        let v_norm = v.iter().fold(H::zero(), |sum, &x| sum + x * x).sqrt();
        if v_norm > NumCast::from(1e-15).unwrap() {
            for i in 0..m - k {
                v[i] /= v_norm;
            }
        }

        // Apply Householder reflection to R: R = (I - 2vv^T)R
        for j in 0..n {
            let mut dot_product = H::zero();
            for i in 0..m - k {
                dot_product += v[i] * r_h[[k + i, j]];
            }

            for i in 0..m - k {
                r_h[[k + i, j]] -= H::from(2.0).unwrap() * v[i] * dot_product;
            }
        }

        // Apply Householder reflection to Q: Q = Q(I - 2vv^T)
        for i in 0..m {
            let mut dot_product = H::zero();
            for j in 0..m - k {
                dot_product += q_h[[i, k + j]] * v[j];
            }

            for j in 0..m - k {
                q_h[[i, k + j]] -= H::from(2.0).unwrap() * dot_product * v[j];
            }
        }
    }

    // Ensure R is upper triangular by setting tiny elements to zero
    for i in 0..m {
        for j in 0..std::cmp::min(i, n) {
            r_h[[i, j]] = H::zero();
        }
    }

    // Ensure Q is orthogonal by normalizing columns
    for j in 0..m {
        let mut col_norm = H::zero();
        for i in 0..m {
            col_norm += q_h[[i, j]] * q_h[[i, j]];
        }
        col_norm = col_norm.sqrt();

        if col_norm > H::from(1e-15).unwrap() {
            for i in 0..m {
                q_h[[i, j]] /= col_norm;
            }
        }
    }

    // Convert back to desired output precision
    let q_c: Array2<C> = convert_2d(&q_h.view());
    let r_c: Array2<C> = convert_2d(&r_h.view());

    Ok((q_c, r_c))
}

/// Perform mixed-precision Singular Value Decomposition (SVD)
///
/// This function computes the SVD of a matrix using a higher precision
/// for internal computations while accepting inputs and producing outputs in specified
/// precisions. The SVD decomposes a matrix A into U * S * V^T, where U and V are
/// orthogonal matrices and S is a diagonal matrix containing singular values.
///
/// # Arguments
///
/// * `a` - Input matrix of precision A
/// * `full_matrices` - If true, return full-sized U and V matrices. Otherwise, economical.
///
/// # Returns
///
/// * Tuple (U, S, V^T) where:
///   - U is an orthogonal matrix of left singular vectors
///   - S is a 1D array of singular values
///   - V^T is an orthogonal matrix of right singular vectors (transposed)
///
/// # Type Parameters
///
/// * `A` - Input matrix precision
/// * `C` - Output matrices/values precision
/// * `H` - Higher precision used for computation
pub fn mixed_precision_svd<A, C, H>(
    a: &ArrayView2<A>,
    full_matrices: bool,
) -> LinalgResult<(Array2<C>, Array1<C>, Array2<C>)>
where
    A: Float + NumAssign + Debug + 'static,
    C: Float + NumAssign + Debug + 'static,
    H: Float + NumAssign + Debug + 'static + std::iter::Sum + ndarray::ScalarOperand,
    A: NumCast,
    C: NumCast,
    H: NumCast,
{
    // Convert input matrix to higher precision
    let a_h: Array2<H> = convert_2d(a);

    // Compute SVD in higher precision
    let (u_h, s_h, vt_h) = svd(&a_h.view(), full_matrices, None)?;

    // Convert back to desired output precision
    let u_c: Array2<C> = convert_2d(&u_h.view());
    let s_c: Array1<C> = convert(&s_h.view());
    let vt_c: Array2<C> = convert_2d(&vt_h.view());

    Ok((u_c, s_c, vt_c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_mixed_precision_solve() {
        let a = array![[2.0f32, 1.0f32], [1.0f32, 3.0f32]];
        let b = array![5.0f32, 8.0f32];

        let x = mixed_precision_solve::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        assert_eq!(x.len(), 2);
        // Verify solution: 2*x[0] + 1*x[1] = 5, 1*x[0] + 3*x[1] = 8
        // Solution should be approximately [1.4, 2.2]
        assert_relative_eq!(x[0], 1.4f32, epsilon = 1e-4);
        assert_relative_eq!(x[1], 2.2f32, epsilon = 1e-4);
    }

    #[test]
    fn test_mixed_precision_cond() {
        // Well-conditioned matrix
        let a = array![[2.0f32, 0.0f32], [0.0f32, 2.0f32]];
        let cond = mixed_precision_cond::<f32, f32, f64>(&a.view(), None).unwrap();
        assert_relative_eq!(cond, 1.0f32, epsilon = 1e-5);

        // Ill-conditioned matrix
        let b = array![[1.0f32, 1.0f32], [1.0f32, 1.0001f32]];
        let cond_b = mixed_precision_cond::<f32, f32, f64>(&b.view(), None).unwrap();
        assert!(cond_b > 1000.0f32); // Should be very large
    }

    #[test]
    fn test_mixed_precision_qr() {
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];

        let (q, r) = mixed_precision_qr::<f32, f32, f64>(&a.view()).unwrap();

        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(r.shape(), &[2, 2]);

        // Verify Q is orthogonal: Q^T * Q ≈ I
        let qt = q.t();
        let qtq = qt.dot(&q);

        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(qtq[[i, j]], expected, epsilon = 1e-4);
            }
        }

        // Verify A = Q * R
        let qr = q.dot(&r);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(qr[[i, j]], a[[i, j]], epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_mixed_precision_svd() {
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];

        let (u, s, vt) = mixed_precision_svd::<f32, f32, f64>(&a.view(), false).unwrap();

        assert_eq!(u.shape()[0], 2);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.shape()[1], 2);

        // Verify singular values are non-negative and sorted
        assert!(s[0] >= s[1]);
        assert!(s[1] >= 0.0);

        // Verify U is orthogonal
        let ut = u.t();
        let uut = ut.dot(&u);
        for i in 0..uut.shape()[0] {
            for j in 0..uut.shape()[1] {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(uut[[i, j]], expected, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_mixed_precision_solve_errors() {
        // Non-square matrix
        let a = array![[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]];
        let b = array![1.0f32, 2.0f32];

        let result = mixed_precision_solve::<f32, f32, f32, f64>(&a.view(), &b.view());
        assert!(result.is_err());

        // Dimension mismatch
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
        let b = array![1.0f32, 2.0f32, 3.0f32];

        let result = mixed_precision_solve::<f32, f32, f32, f64>(&a.view(), &b.view());
        assert!(result.is_err());

        // Singular matrix
        let a = array![[1.0f32, 2.0f32], [2.0f32, 4.0f32]];
        let b = array![1.0f32, 2.0f32];

        let result = mixed_precision_solve::<f32, f32, f32, f64>(&a.view(), &b.view());
        assert!(result.is_err());
    }
}
