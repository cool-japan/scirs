//! Advanced matrix decomposition algorithms
//!
//! This module provides specialized decomposition algorithms that complement
//! the standard decompositions, focusing on higher accuracy or specific use cases.

use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};
use crate::norm::matrix_norm;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign};
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
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::decomposition_advanced::jacobi_svd;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let (u, s, vt) = jacobi_svd(&a.view(), 100, 1e-14).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn jacobi_svd<A>(
    a: &ArrayView2<A>,
    max_iterations: usize,
    tolerance: A,
) -> LinalgResult<(Array2<A>, Array1<A>, Array2<A>)>
where
    A: Float
        + NumAssign
        + Debug
        + Display
        + scirs2_core::ndarray::ScalarOperand
        + std::iter::Sum
        + Send
        + Sync
        + 'static,
{
    let (m, n) = (a.nrows(), a.ncols());

    // For wide matrices (m < n): transpose, delegate to tall path, swap U<->V.
    if m < n {
        let at = a.t().to_owned();
        // at is n×m (tall), SVD: at = U_t · diag(s) · Vt_t
        // => a = Vt_t^T · diag(s) · U_t^T
        let (u_t, s, vt_t) = jacobi_svd(&at.view(), max_iterations, tolerance)?;
        // U of a = Vt_t^T (shape n→m for wide)... but we want thin:
        // U (m×m), s (m), Vt (m×n)
        // u_t is n×m, vt_t is m×m
        // U of a = Vt_t^T → m×m;  Vt of a = U_t^T → m×n
        let u_of_a = vt_t.t().to_owned();
        let vt_of_a = u_t.t().to_owned();
        return Ok((u_of_a, s, vt_of_a));
    }

    // For m >= n (square or tall matrix).
    // Strategy: use normal equations A^T·A (n×n symmetric PSD) for the tall case.
    // The two-sided symmetric Jacobi diagonalizes symmetric matrices correctly.
    // A^T·A = V · diag(s²) · V^T  →  A = U · diag(s) · V^T with U = A·V·diag(1/s).
    let (b_sym, is_tall) = if m == n {
        // Square: apply Jacobi directly to A.
        (a.to_owned(), false)
    } else {
        // Tall (m > n): form n×n Gram matrix A^T·A.
        let ata = a.t().dot(a);
        (ata, true)
    };

    // ---- two-sided symmetric Jacobi on b_sym (always n×n) ----
    let sq = b_sym.nrows(); // == n
                            // Initialize U and V as identity matrices
    let mut u = Array2::eye(sq);
    let mut v = Array2::eye(sq);
    let mut b = b_sym;

    // Jacobi rotation iterations — b and u/v are all sq×sq.
    for _iter in 0..max_iterations {
        let mut max_off_diag = A::zero();
        let mut p = 0;
        let mut q_idx = 0;

        // Find the largest off-diagonal element
        for i in 0..sq {
            for j in (i + 1)..sq {
                let val = b[[i, j]].abs() + b[[j, i]].abs();
                if val > max_off_diag {
                    max_off_diag = val;
                    p = i;
                    q_idx = j;
                }
            }
        }

        // Check convergence
        if max_off_diag < tolerance {
            break;
        }

        // Compute the rotation angle
        let app = b[[p, p]];
        let aqq = b[[q_idx, q_idx]];
        let apq = b[[p, q_idx]];
        let aqp = b[[q_idx, p]];

        // For symmetric part
        let theta = if (app - aqq).abs() < A::epsilon() {
            A::from(std::f64::consts::PI / 4.0).ok_or_else(|| {
                LinalgError::ComputationError("Float conversion failed".to_string())
            })?
        } else {
            ((apq + aqp) / (app - aqq)).atan()
                * A::from(0.5).ok_or_else(|| {
                    LinalgError::ComputationError("Float conversion failed".to_string())
                })?
        };

        let c = theta.cos();
        let s_rot = theta.sin();

        // Apply Givens rotation from left: B' = G^T * B
        for j in 0..sq {
            let bpj = b[[p, j]];
            let bqj = b[[q_idx, j]];
            b[[p, j]] = c * bpj + s_rot * bqj;
            b[[q_idx, j]] = -s_rot * bpj + c * bqj;
        }

        // Apply Givens rotation from right: B'' = B' * G
        for i in 0..sq {
            let bip = b[[i, p]];
            let biq = b[[i, q_idx]];
            b[[i, p]] = c * bip + s_rot * biq;
            b[[i, q_idx]] = -s_rot * bip + c * biq;
        }

        // Update U_sq: U' = U * G
        for i in 0..sq {
            let uip = u[[i, p]];
            let uiq = u[[i, q_idx]];
            u[[i, p]] = c * uip + s_rot * uiq;
            u[[i, q_idx]] = -s_rot * uip + c * uiq;
        }

        // Update V: V' = V * G
        for i in 0..sq {
            let vip = v[[i, p]];
            let viq = v[[i, q_idx]];
            v[[i, p]] = c * vip + s_rot * viq;
            v[[i, q_idx]] = -s_rot * vip + c * viq;
        }
    }

    // Extract singular values from the diagonal of the (now near-diagonal) b matrix.
    // For the square case, b_ii are the singular values directly.
    // For the tall case (Gram matrix A^T·A), b_ii are eigenvalues (s²), so take sqrt.
    let mut s_vec = Array1::zeros(sq);
    for i in 0..sq {
        let raw = b[[i, i]].abs();
        s_vec[i] = if is_tall { raw.sqrt() } else { raw };
    }

    // Sort singular values in descending order.
    let mut indices: Vec<usize> = (0..sq).collect();
    indices.sort_by(|&i, &j| {
        s_vec[j]
            .partial_cmp(&s_vec[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Reorder singular values, U_sq, and V.
    let mut s_sorted = Array1::zeros(sq);
    let mut u_sq_sorted = Array2::zeros((sq, sq));
    let mut v_sorted = Array2::zeros((sq, sq));

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        s_sorted[new_idx] = s_vec[old_idx];
        for i in 0..sq {
            u_sq_sorted[[i, new_idx]] = u[[i, old_idx]];
            v_sorted[[i, new_idx]] = v[[i, old_idx]];
        }
    }

    // For the symmetric Gram-matrix (tall) path, the "U" from Jacobi is not U of A.
    // We accumulated V (right singular vectors of A^T·A = right singular vectors of A).
    // Singular values of A are sqrt of eigenvalues of A^T·A.
    // Left singular vectors: u_i = A · v_i / s_i.
    let u_final = if is_tall {
        // v_sorted columns are the right singular vectors of A.
        // Compute U = A · V · diag(1/s)
        let mut u_lift = Array2::zeros((m, sq));
        for j in 0..sq {
            let sj = s_sorted[j];
            if sj > A::epsilon() {
                let vj = v_sorted.column(j);
                let av_j = a.dot(&vj);
                for i in 0..m {
                    u_lift[[i, j]] = av_j[i] / sj;
                }
            }
            // If s_j ≈ 0, leave the corresponding column as zero (orthogonalize later
            // if needed, but for reconstruction accuracy it doesn't matter).
        }
        // Convert s_sorted from sqrt-eigenvalue to actual singular values.
        // (They already are: we took sqrt below when extracting from A^T·A diagonal.)
        u_lift
    } else {
        // Square case: U is directly u_sq_sorted.
        u_sq_sorted
    };

    Ok((u_final, s_sorted, v_sorted.t().to_owned()))
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
/// * Tuple `(U, Option<P>)` where U is unitary and P is positive semidefinite
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::decomposition_advanced::polar_decomposition;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let (u, p) = polar_decomposition(&a.view(), true).expect("Operation failed");
/// assert!(p.is_some());
/// ```
#[allow(dead_code)]
pub fn polar_decomposition<A>(
    a: &ArrayView2<A>,
    compute_p: bool,
) -> LinalgResult<(Array2<A>, Option<Array2<A>>)>
where
    A: Float
        + NumAssign
        + Debug
        + Display
        + scirs2_core::ndarray::ScalarOperand
        + std::iter::Sum
        + Send
        + Sync
        + 'static,
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
#[allow(dead_code)]
pub fn polar_decomposition_newton<A>(
    a: &ArrayView2<A>,
    max_iterations: usize,
    tolerance: A,
) -> LinalgResult<(Array2<A>, Array2<A>)>
where
    A: Float
        + NumAssign
        + Debug
        + Display
        + scirs2_core::ndarray::ScalarOperand
        + std::iter::Sum
        + Send
        + Sync
        + 'static,
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

        let u_new = (&u + &ut_inv) * A::from(0.5).expect("Operation failed");

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
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::decomposition_advanced::qr_with_column_pivoting;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let (q, r, p, rank) = qr_with_column_pivoting(&a.view(), 1e-10).expect("Operation failed");
/// assert!(rank < 3); // Matrix is rank-deficient
/// ```
pub type QRPivotingResult<A> = (Array2<A>, Array2<A>, Array2<A>, usize);

#[allow(dead_code)]
pub fn qr_with_column_pivoting<A>(
    a: &ArrayView2<A>,
    tolerance: A,
) -> LinalgResult<QRPivotingResult<A>>
where
    A: Float
        + NumAssign
        + Debug
        + Display
        + scirs2_core::ndarray::ScalarOperand
        + std::iter::Sum
        + Send
        + Sync
        + 'static,
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
    use scirs2_core::ndarray::array;

    /// Helper: verify A == U * diag(s) * Vt within `eps`.
    fn check_svd_reconstruction(
        a: &scirs2_core::ndarray::ArrayView2<f64>,
        u: &Array2<f64>,
        s: &Array1<f64>,
        vt: &Array2<f64>,
        eps: f64,
    ) {
        let s_diag = Array2::from_diag(s);
        let reconstructed = u.dot(&s_diag).dot(vt);
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_abs_diff_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = eps);
            }
        }
    }

    #[test]
    fn test_jacobi_svd_3x2_tall() {
        // Tall 3×2 matrix — exercises QR-then-Jacobi path.
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (u, s, vt) = jacobi_svd(&a.view(), 200, 1e-12).expect("jacobi_svd 3x2 failed");

        // Dimensions: U is 3×2, s has 2 values, Vt is 2×2.
        assert_eq!(u.shape(), &[3, 2]);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.shape(), &[2, 2]);

        // Singular values must be positive and sorted.
        assert!(s[0] >= s[1] && s[1] >= 0.0);

        // U must have orthonormal columns: U^T U = I_2.
        let utu = u.t().dot(&u);
        assert_abs_diff_eq!(utu[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(utu[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(utu[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(utu[[1, 1]], 1.0, epsilon = 1e-10);

        // Vt must be orthogonal: Vt Vt^T = I_2.
        let vvt = vt.dot(&vt.t());
        assert_abs_diff_eq!(vvt[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(vvt[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(vvt[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(vvt[[1, 1]], 1.0, epsilon = 1e-10);

        // Reconstruction: A == U diag(s) Vt.
        check_svd_reconstruction(&a.view(), &u, &s, &vt, 1e-10);
    }

    #[test]
    fn test_jacobi_svd_2x3_wide() {
        // Wide 2×3 matrix — exercises transpose-delegate path.
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let (u, s, vt) = jacobi_svd(&a.view(), 200, 1e-12).expect("jacobi_svd 2x3 failed");

        // Dimensions: U is 2×2, s has 2 values, Vt is 2×3.
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.shape(), &[2, 3]);

        // Singular values must be positive and sorted.
        assert!(s[0] >= s[1] && s[1] >= 0.0);

        // U orthogonality.
        let utu = u.t().dot(&u);
        assert_abs_diff_eq!(utu[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(utu[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(utu[[0, 1]], 0.0, epsilon = 1e-10);

        // Reconstruction.
        check_svd_reconstruction(&a.view(), &u, &s, &vt, 1e-10);
    }

    #[test]
    fn test_jacobi_svd_2x2() {
        let a = array![[3.0, 1.0], [1.0, 3.0]];
        let (u, s, vt) = jacobi_svd(&a.view(), 100, 1e-14).expect("Operation failed");

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
        let (u, p_opt) = polar_decomposition(&a.view(), true).expect("Operation failed");
        let p = p_opt.expect("Operation failed");

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
        let (u, p) = polar_decomposition_newton(&a.view(), 10, 1e-12).expect("Operation failed");

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
        let (q, r, p, rank) = qr_with_column_pivoting(&a.view(), 1e-10).expect("Operation failed");

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
