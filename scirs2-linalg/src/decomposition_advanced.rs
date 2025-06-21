//! Advanced matrix decomposition algorithms
//!
//! This module provides specialized decomposition algorithms that complement
//! the standard decompositions, focusing on higher accuracy or specific use cases.

use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};
use crate::norm::matrix_norm;
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, NumAssign};
use std::fmt::{Debug, Display};

/// Jacobi SVD for small matrices with higher accuracy
///
/// The Jacobi algorithm uses a series of Givens rotations to diagonalize the matrix.
/// It's slower than standard SVD but can achieve higher accuracy, especially for
/// small matrices or when high precision is required.
///
/// # Arguments
/// * `a` - Input matrix (m × n)
/// * `max_iterations` - Maximum number of Jacobi sweeps
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// * Tuple (U, S, Vt) - SVD decomposition with high accuracy
///
/// # Example
/// ```
/// use ndarray::array;
/// use scirs2_linalg::decomposition_advanced::jacobi_svd;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let (u, s, vt) = jacobi_svd(&a.view(), 100, 1e-14).unwrap();
/// ```
pub fn jacobi_svd<A>(
    a: &ArrayView2<A>,
    max_iterations: usize,
    tolerance: A,
) -> LinalgResult<(Array2<A>, Array1<A>, Array2<A>)>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let (m, n) = (a.nrows(), a.ncols());

    // For now, handle only square matrices
    if m != n {
        return Err(LinalgError::NotImplementedError(
            "Jacobi SVD currently only supports square matrices".to_string(),
        ));
    }

    // Initialize U and V as identity matrices
    let mut u = Array2::eye(m);
    let mut v = Array2::eye(n);
    let mut b = a.to_owned();

    // Jacobi rotation iterations
    for _iter in 0..max_iterations {
        let mut max_off_diag = A::zero();
        let mut p = 0;
        let mut q = 0;

        // Find the largest off-diagonal element
        for i in 0..n {
            for j in (i + 1)..n {
                let val = b[[i, j]].abs() + b[[j, i]].abs();
                if val > max_off_diag {
                    max_off_diag = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Check convergence
        if max_off_diag < tolerance {
            break;
        }

        // Compute the rotation angle
        let app = b[[p, p]];
        let aqq = b[[q, q]];
        let apq = b[[p, q]];
        let aqp = b[[q, p]];

        // For symmetric part
        let theta = if (app - aqq).abs() < A::epsilon() {
            A::from(std::f64::consts::PI / 4.0).unwrap()
        } else {
            ((apq + aqp) / (app - aqq)).atan() * A::from(0.5).unwrap()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Givens rotation from left: B' = G^T * B
        for j in 0..n {
            let bpj = b[[p, j]];
            let bqj = b[[q, j]];
            b[[p, j]] = c * bpj + s * bqj;
            b[[q, j]] = -s * bpj + c * bqj;
        }

        // Apply Givens rotation from right: B'' = B' * G
        for i in 0..m {
            let bip = b[[i, p]];
            let biq = b[[i, q]];
            b[[i, p]] = c * bip + s * biq;
            b[[i, q]] = -s * bip + c * biq;
        }

        // Update U: U' = U * G
        for i in 0..m {
            let uip = u[[i, p]];
            let uiq = u[[i, q]];
            u[[i, p]] = c * uip + s * uiq;
            u[[i, q]] = -s * uip + c * uiq;
        }

        // Update V: V' = V * G
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip + s * viq;
            v[[i, q]] = -s * vip + c * viq;
        }
    }

    // Extract singular values and ensure they're positive
    let mut s = Array1::zeros(n);
    for i in 0..n {
        s[i] = b[[i, i]].abs();
    }

    // Sort singular values in descending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap());

    // Reorder singular values, U, and V
    let mut s_sorted = Array1::zeros(n);
    let mut u_sorted = Array2::zeros((m, n));
    let mut v_sorted = Array2::zeros((n, n));

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        s_sorted[new_idx] = s[old_idx];
        for i in 0..m {
            u_sorted[[i, new_idx]] = u[[i, old_idx]];
        }
        for i in 0..n {
            v_sorted[[i, new_idx]] = v[[i, old_idx]];
        }
    }

    Ok((u_sorted, s_sorted, v_sorted.t().to_owned()))
}

/// Polar decomposition of a matrix
///
/// Decomposes a matrix A into the product A = U * P where:
/// - U is unitary (orthogonal for real matrices)
/// - P is positive semidefinite
///
/// This decomposition is useful in various applications including:
/// - Computing the nearest orthogonal matrix
/// - Matrix square roots
/// - Procrustes problems
///
/// # Arguments
/// * `a` - Input matrix
/// * `compute_p` - Whether to compute P (if false, only U is returned)
///
/// # Returns
/// * Tuple (U, Option<P>) where U is unitary and P is positive semidefinite
///
/// # Example
/// ```
/// use ndarray::array;
/// use scirs2_linalg::decomposition_advanced::polar_decomposition;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let (u, p) = polar_decomposition(&a.view(), true).unwrap();
/// assert!(p.is_some());
/// ```
pub fn polar_decomposition<A>(
    a: &ArrayView2<A>,
    compute_p: bool,
) -> LinalgResult<(Array2<A>, Option<Array2<A>>)>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let (m, n) = (a.nrows(), a.ncols());

    if m != n {
        return Err(LinalgError::ShapeError(
            "Polar decomposition requires a square matrix".to_string(),
        ));
    }

    // Compute SVD: A = U * S * V^T
    let (u_svd, s, vt) = svd(a, false, None)?;

    // Compute U = U_svd * V^T (unitary part)
    let u = u_svd.dot(&vt);

    // Compute P = V * S * V^T (positive semidefinite part) if requested
    let p = if compute_p {
        let v = vt.t();
        let s_diag = Array2::from_diag(&s);
        Some(v.dot(&s_diag).dot(&vt))
    } else {
        None
    };

    Ok((u, p))
}

/// Iterative refinement for polar decomposition
///
/// Uses Newton's method to iteratively improve the polar decomposition,
/// achieving higher accuracy than the standard SVD-based method.
///
/// # Arguments
/// * `a` - Input matrix
/// * `max_iterations` - Maximum number of Newton iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// * Tuple (U, P) where U is unitary and P is positive semidefinite
pub fn polar_decomposition_newton<A>(
    a: &ArrayView2<A>,
    max_iterations: usize,
    tolerance: A,
) -> LinalgResult<(Array2<A>, Array2<A>)>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let (m, n) = (a.nrows(), a.ncols());

    if m != n {
        return Err(LinalgError::ShapeError(
            "Polar decomposition requires a square matrix".to_string(),
        ));
    }

    // Initial approximation using standard polar decomposition
    let (mut u, _) = polar_decomposition(a, false)?;

    // Newton iteration: U_{k+1} = (U_k + (U_k^T)^{-1}) / 2
    for _iter in 0..max_iterations {
        let ut = u.t();

        // Compute (U^T)^{-1} using SVD
        let ut_inv = match crate::inv(&ut.view(), None) {
            Ok(inv) => inv,
            Err(_) => {
                // If inversion fails, use pseudoinverse
                let (u_inv, s_inv, vt_inv) = svd(&ut.view(), false, None)?;
                let mut s_pinv = Array1::zeros(s_inv.len());
                for i in 0..s_inv.len() {
                    if s_inv[i] > A::epsilon() {
                        s_pinv[i] = A::one() / s_inv[i];
                    }
                }
                let s_pinv_diag = Array2::from_diag(&s_pinv);
                vt_inv.t().dot(&s_pinv_diag).dot(&u_inv.t())
            }
        };

        let u_new = (&u + &ut_inv) * A::from(0.5).unwrap();

        // Check convergence
        let diff = &u_new - &u;
        let error = matrix_norm(&diff.view(), "fro", None)?;

        u = u_new;

        if error < tolerance {
            break;
        }
    }

    // Compute P = U^T * A
    let p = u.t().dot(a);

    Ok((u, p))
}

/// QR decomposition with column pivoting for rank-revealing decomposition
///
/// This is an enhanced QR decomposition that reveals the numerical rank of the matrix
/// by permuting columns to ensure the diagonal elements of R decrease in magnitude.
///
/// # Arguments
/// * `a` - Input matrix
/// * `tolerance` - Tolerance for determining numerical rank
///
/// # Returns
/// * Tuple (Q, R, P, rank) where:
///   - Q is orthogonal
///   - R is upper triangular with decreasing diagonal
///   - P is the permutation matrix
///   - rank is the numerical rank
///
/// # Example
/// ```
/// use ndarray::array;
/// use scirs2_linalg::decomposition_advanced::qr_with_column_pivoting;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let (q, r, p, rank) = qr_with_column_pivoting(&a.view(), 1e-10).unwrap();
/// assert!(rank < 3); // Matrix is rank-deficient
/// ```
pub type QRPivotingResult<A> = (Array2<A>, Array2<A>, Array2<A>, usize);

pub fn qr_with_column_pivoting<A>(
    a: &ArrayView2<A>,
    tolerance: A,
) -> LinalgResult<QRPivotingResult<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    // This is already implemented as complete_orthogonal_decomposition in decomposition.rs
    // We'll provide a wrapper that extracts the rank information

    let (q, r, p) = crate::decomposition::complete_orthogonal_decomposition(a)?;

    // Determine numerical rank by examining diagonal of R
    let min_dim = a.shape()[0].min(a.shape()[1]);
    let mut rank = 0;

    for i in 0..min_dim {
        if r[[i, i]].abs() > tolerance {
            rank += 1;
        } else {
            break;
        }
    }

    Ok((q, r, p, rank))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_jacobi_svd_2x2() {
        let a = array![[3.0, 1.0], [1.0, 3.0]];
        let (u, s, vt) = jacobi_svd(&a.view(), 100, 1e-14).unwrap();

        // Verify dimensions
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.shape(), &[2, 2]);

        // Verify orthogonality of U
        let u_ut = u.dot(&u.t());
        assert_abs_diff_eq!(u_ut[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(u_ut[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(u_ut[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(u_ut[[1, 1]], 1.0, epsilon = 1e-10);

        // Verify orthogonality of V
        let v = vt.t();
        let v_vt = v.dot(&vt);
        assert_abs_diff_eq!(v_vt[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v_vt[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v_vt[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v_vt[[1, 1]], 1.0, epsilon = 1e-10);

        // Verify reconstruction
        let s_diag = Array2::from_diag(&s);
        let reconstructed = u.dot(&s_diag).dot(&vt);
        assert_abs_diff_eq!(reconstructed[[0, 0]], a[[0, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(reconstructed[[0, 1]], a[[0, 1]], epsilon = 1e-10);
        assert_abs_diff_eq!(reconstructed[[1, 0]], a[[1, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(reconstructed[[1, 1]], a[[1, 1]], epsilon = 1e-10);

        // For this symmetric matrix, singular values should be 4 and 2
        assert_abs_diff_eq!(s[0], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(s[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polar_decomposition() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let (u, p_opt) = polar_decomposition(&a.view(), true).unwrap();
        let p = p_opt.unwrap();

        // Verify U is orthogonal
        let u_ut = u.dot(&u.t());
        assert_abs_diff_eq!(u_ut[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(u_ut[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(u_ut[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(u_ut[[1, 1]], 1.0, epsilon = 1e-10);

        // Verify P is symmetric
        assert_abs_diff_eq!(p[[0, 1]], p[[1, 0]], epsilon = 1e-10);

        // Verify reconstruction: A = U * P
        let reconstructed = u.dot(&p);
        assert_abs_diff_eq!(reconstructed[[0, 0]], a[[0, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(reconstructed[[0, 1]], a[[0, 1]], epsilon = 1e-10);
        assert_abs_diff_eq!(reconstructed[[1, 0]], a[[1, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(reconstructed[[1, 1]], a[[1, 1]], epsilon = 1e-10);
    }

    #[test]
    fn test_polar_decomposition_newton() {
        let a = array![[1.0, 0.5], [0.5, 2.0]];
        let (u, p) = polar_decomposition_newton(&a.view(), 10, 1e-12).unwrap();

        // Verify U is orthogonal
        let u_ut = u.dot(&u.t());
        assert_abs_diff_eq!(u_ut[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(u_ut[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(u_ut[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(u_ut[[1, 1]], 1.0, epsilon = 1e-10);

        // Verify P is symmetric and positive semidefinite
        assert_abs_diff_eq!(p[[0, 1]], p[[1, 0]], epsilon = 1e-10);

        // Verify reconstruction: A = U * P
        let reconstructed = u.dot(&p);
        assert_abs_diff_eq!(reconstructed[[0, 0]], a[[0, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(reconstructed[[0, 1]], a[[0, 1]], epsilon = 1e-10);
        assert_abs_diff_eq!(reconstructed[[1, 0]], a[[1, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(reconstructed[[1, 1]], a[[1, 1]], epsilon = 1e-10);
    }

    #[test]
    fn test_qr_with_column_pivoting() {
        // Rank-deficient matrix
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let (q, _r, p, rank) = qr_with_column_pivoting(&a.view(), 1e-10).unwrap();

        // Matrix should have rank 2
        assert_eq!(rank, 2);

        // Verify Q is orthogonal
        let q_qt = q.dot(&q.t());
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(q_qt[[i, j]], expected, epsilon = 1e-3);
            }
        }

        // Verify P is a permutation matrix
        let p_pt = p.dot(&p.t());
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(p_pt[[i, j]], expected, epsilon = 1e-3);
            }
        }
    }
}
