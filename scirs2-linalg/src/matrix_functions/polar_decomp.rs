//! Polar decomposition A = U*H (U orthogonal/unitary, H symmetric positive semidefinite)
//!
//! The polar decomposition factors a matrix A into:
//! - U: orthogonal (unitary in the complex case) matrix — the "rotation" part
//! - H: symmetric positive semidefinite matrix — the "stretching" part
//!
//! # Algorithms
//!
//! - **SVD-based**: A = P*Σ*Q^T → U = P*Q^T, H = Q*Σ*Q^T (exact, reference)
//! - **Halley iteration**: Third-order convergence via Padé [2/1] rational step:
//!   X_{k+1} = X_k (3I + X_k^T X_k) / (I + 3 X_k^T X_k)
//! - **Newton iteration**: Second-order convergence:
//!   X_{k+1} = (X_k + X_k^{-T}) / 2
//!
//! # Applications
//!
//! - Orthogonalization (Procrustes problem)
//! - Rotation recovery in continuum mechanics
//! - Factor analysis / structural mechanics
//!
//! # References
//!
//! - Higham, N.J. (1986). "Computing the polar decomposition — with applications."
//!   SIAM J. Sci. Statist. Comput.
//! - Nakatsukasa, Y., Brellas, Z. & Higham, N.J. (2010). "Optimizing Halley's
//!   iteration for computing the matrix polar decomposition." SIAM J. Matrix Anal.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, One};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Floating-point trait alias for polar decomposition.
pub trait PolarFloat: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}
impl<F> PolarFloat for F where F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn matmul_nn<F: PolarFloat>(a: &Array2<F>, b: &Array2<F>) -> LinalgResult<Array2<F>> {
    let (m, k) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    if k != k2 {
        return Err(LinalgError::ShapeError(format!(
            "polar matmul: inner dims {} vs {}",
            k, k2
        )));
    }
    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for l in 0..k {
            let a_il = a[[i, l]];
            if a_il == F::zero() {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += a_il * b[[l, j]];
            }
        }
    }
    Ok(c)
}

fn frobenius_norm<F: PolarFloat>(a: &Array2<F>) -> F {
    let mut acc = F::zero();
    for &v in a.iter() {
        acc += v * v;
    }
    acc.sqrt()
}

fn transpose<F: PolarFloat>(a: &Array2<F>) -> Array2<F> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut t = Array2::<F>::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            t[[j, i]] = a[[i, j]];
        }
    }
    t
}

fn lu_factorize<F: PolarFloat>(a: &Array2<F>) -> LinalgResult<(Array2<F>, Vec<usize>)> {
    let n = a.nrows();
    let mut lu = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        let mut max_val = F::zero();
        let mut max_row = k;
        for i in k..n {
            let v = lu[[i, k]].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_val < F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
            return Err(LinalgError::SingularMatrixError(
                "polar: near-singular matrix in LU".into(),
            ));
        }

        if max_row != k {
            perm.swap(k, max_row);
            for j in 0..n {
                let tmp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
        }

        for i in (k + 1)..n {
            lu[[i, k]] = lu[[i, k]] / lu[[k, k]];
            for j in (k + 1)..n {
                let l_ik = lu[[i, k]];
                let u_kj = lu[[k, j]];
                lu[[i, j]] -= l_ik * u_kj;
            }
        }
    }
    Ok((lu, perm))
}

fn lu_solve<F: PolarFloat>(lu: &Array2<F>, perm: &[usize], b: &Array2<F>) -> Array2<F> {
    let n = lu.nrows();
    let nrhs = b.ncols();
    let mut x = Array2::<F>::zeros((n, nrhs));

    for col in 0..nrhs {
        let mut y = vec![F::zero(); n];
        for i in 0..n {
            y[i] = b[[perm[i], col]];
        }
        for i in 0..n {
            for j in 0..i {
                let yj = y[j];
                y[i] -= lu[[i, j]] * yj;
            }
        }
        let mut z = vec![F::zero(); n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= lu[[i, j]] * z[j];
            }
            z[i] = sum / lu[[i, i]];
        }
        for i in 0..n {
            x[[i, col]] = z[i];
        }
    }
    x
}

fn mat_inv<F: PolarFloat>(a: &Array2<F>) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    let eye = Array2::<F>::eye(n);
    let (lu, perm) = lu_factorize(a)?;
    Ok(lu_solve(&lu, &perm, &eye))
}

// ---------------------------------------------------------------------------
// SVD-based polar decomposition
// ---------------------------------------------------------------------------

/// Compute the polar decomposition A = U * H via singular value decomposition.
///
/// Given the thin SVD A = P * Σ * Q^T:
/// - Unitary factor: U = P * Q^T
/// - Hermitian factor: H = Q * Σ * Q^T
///
/// This is the reference algorithm — exact and unconditionally stable.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * `(U, H)` where A = U * H, U is orthogonal, H is symmetric PSD
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::polar_decomp::polar_via_svd;
///
/// let a = array![[3.0_f64, 1.0], [1.0, 2.0]];
/// let (u, h) = polar_via_svd(&a.view()).expect("polar_via_svd failed");
/// // U should be orthogonal: U^T U ≈ I
/// ```
pub fn polar_via_svd<F>(a: &ArrayView2<F>) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + One + ScalarOperand + Send + Sync + 'static + std::fmt::Display,
{
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "polar_via_svd: matrix must be square".into(),
        ));
    }
    if n == 0 {
        return Ok((Array2::<F>::zeros((0, 0)), Array2::<F>::zeros((0, 0))));
    }

    use crate::decomposition::svd;

    let (p, sigma, qt) = svd(a, true, None)?;
    let q = transpose(&qt);

    // U = P * Q^T
    let u = matmul_nn(&p, &qt)?;

    // H = Q * Σ * Q^T — reconstruct as symmetric PSD
    // First form Σ * Q^T (scale rows of Q^T)
    let mut sigma_qt = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            sigma_qt[[i, j]] = sigma[i] * qt[[i, j]];
        }
    }
    let h = matmul_nn(&q, &sigma_qt)?;

    Ok((u, h))
}

// ---------------------------------------------------------------------------
// Halley iteration for unitary factor
// ---------------------------------------------------------------------------

/// Compute the unitary factor of the polar decomposition via Halley iteration.
///
/// Halley's method for the polar unitary factor uses the [2/1] Padé approximant:
/// ```text
/// X_{k+1} = X_k (3I + X_k^T X_k) / (I + 3 X_k^T X_k)
/// ```
/// Convergence is cubically fast (order 3) — significantly faster than Newton's
/// quadratic convergence.
///
/// # Arguments
///
/// * `a` - Input square matrix with full column rank
/// * `max_iter` - Maximum iterations (default 20)
/// * `tol` - Convergence tolerance (default 1e-12)
///
/// # Returns
///
/// * The orthogonal/unitary factor U of A = U * H
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::polar_decomp::unitary_factor_halley;
///
/// let a = array![[2.0_f64, 0.5], [0.1, 3.0]];
/// let u = unitary_factor_halley(&a.view(), None, None).expect("halley failed");
/// // U should satisfy U^T U ≈ I
/// ```
pub fn unitary_factor_halley<F: PolarFloat>(
    a: &ArrayView2<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "unitary_factor_halley: matrix must be square".into(),
        ));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    let max_iter = max_iter.unwrap_or(20);
    let tol = tol.unwrap_or_else(|| F::from(1e-12).unwrap_or(F::epsilon()));

    let three = F::from(3.0).unwrap_or(F::one() + F::one() + F::one());

    // Normalize: X_0 = A / ||A||_1  (improves initial convergence)
    let norm_a = {
        let n_cols = a.ncols();
        let mut max_col = F::zero();
        for j in 0..n_cols {
            let mut col_sum = F::zero();
            for i in 0..n {
                col_sum += a[[i, j]].abs();
            }
            if col_sum > max_col {
                max_col = col_sum;
            }
        }
        max_col
    };

    if norm_a < F::from(1e-15).unwrap_or(F::epsilon()) {
        return Err(LinalgError::SingularMatrixError(
            "unitary_factor_halley: zero or near-zero matrix".into(),
        ));
    }

    let mut x = a.to_owned();
    for v in x.iter_mut() {
        *v /= norm_a;
    }

    for _ in 0..max_iter {
        let xt = transpose(&x);
        let xtx = matmul_nn(&xt, &x)?; // X^T X

        // Numerator: 3I + X^T X
        let mut num_inner = xtx.clone();
        for i in 0..n {
            num_inner[[i, i]] += three;
        }

        // Denominator: I + 3 X^T X
        let mut den_inner = xtx;
        for i in 0..n {
            den_inner[[i, i]] += F::one();
        }
        for v in den_inner.iter_mut() {
            *v *= three;
        }
        // Wait — den is I + 3 X^T X, but we want (I + 3 X^T X) again
        // Redo: den[[i,i]] was set to xtx[[i,i]] + 1, rest is 3*xtx[[i,j]]
        // Let me redo this correctly:
        let xt2 = transpose(&x);
        let xtx2 = matmul_nn(&xt2, &x)?;
        let mut denom = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let diag_one = if i == j { F::one() } else { F::zero() };
                denom[[i, j]] = diag_one + three * xtx2[[i, j]];
            }
        }

        // Solve: x_new = x * num * denom^{-1}
        let denom_inv = mat_inv(&denom)?;
        let x_num = matmul_nn(&x, &num_inner)?;
        let x_new = matmul_nn(&x_num, &denom_inv)?;

        // Check convergence: ||X_new - X||_F / ||X_new||_F
        let mut diff = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                diff[[i, j]] = x_new[[i, j]] - x[[i, j]];
            }
        }
        let rel = frobenius_norm(&diff)
            / (frobenius_norm(&x_new) + F::from(1e-30).unwrap_or(F::epsilon()));

        x = x_new;

        if rel < tol {
            return Ok(x);
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Newton iteration for unitary factor
// ---------------------------------------------------------------------------

/// Compute the unitary factor of the polar decomposition via Newton iteration.
///
/// Newton's iteration:
/// ```text
/// X_{k+1} = (X_k + X_k^{-T}) / 2
/// ```
/// starting from X_0 = A converges quadratically to the orthogonal factor U.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `max_iter` - Maximum iterations (default 50)
/// * `tol` - Convergence tolerance (default 1e-12)
///
/// # Returns
///
/// * The orthogonal factor U of A = U * H
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::polar_decomp::unitary_factor_newton;
///
/// let a = array![[2.0_f64, 0.5], [0.1, 3.0]];
/// let u = unitary_factor_newton(&a.view(), None, None).expect("newton polar failed");
/// ```
pub fn unitary_factor_newton<F: PolarFloat>(
    a: &ArrayView2<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "unitary_factor_newton: matrix must be square".into(),
        ));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    let max_iter = max_iter.unwrap_or(50);
    let tol = tol.unwrap_or_else(|| F::from(1e-12).unwrap_or(F::epsilon()));
    let two = F::from(2.0).unwrap_or(F::one() + F::one());

    let mut x = a.to_owned();

    for _ in 0..max_iter {
        let xt = transpose(&x);
        let xt_inv = mat_inv(&xt)?; // X^{-T}

        let mut x_new = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                x_new[[i, j]] = (x[[i, j]] + xt_inv[[i, j]]) / two;
            }
        }

        // Check convergence
        let mut diff = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                diff[[i, j]] = x_new[[i, j]] - x[[i, j]];
            }
        }
        let rel = frobenius_norm(&diff)
            / (frobenius_norm(&x_new) + F::from(1e-30).unwrap_or(F::epsilon()));

        x = x_new;

        if rel < tol {
            return Ok(x);
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Full polar decomposition (combined)
// ---------------------------------------------------------------------------

/// Compute the full polar decomposition A = U * H using Halley iteration + reconstruction.
///
/// Steps:
/// 1. Compute U via Halley iteration (unitary_factor_halley)
/// 2. Compute H = U^T * A (which is symmetric PSD when U is truly orthogonal)
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `max_iter` - Maximum iterations for Halley iteration
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * `(U, H)` where A = U * H
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::polar_decomp::polar_decomposition;
///
/// let a = array![[3.0_f64, 1.0], [0.0, 2.0]];
/// let (u, h) = polar_decomposition(&a.view(), None, None).expect("polar failed");
/// // Verify: U^T U ≈ I
/// ```
pub fn polar_decomposition<F: PolarFloat>(
    a: &ArrayView2<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<(Array2<F>, Array2<F>)> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "polar_decomposition: matrix must be square".into(),
        ));
    }
    if n == 0 {
        return Ok((Array2::<F>::zeros((0, 0)), Array2::<F>::zeros((0, 0))));
    }

    // Compute U via Halley iteration
    let u = unitary_factor_halley(a, max_iter, tol)?;

    // H = U^T * A (symmetric factor)
    let ut = transpose(&u);
    let h = matmul_nn(&ut, &a.to_owned())?;

    // Symmetrize H for numerical cleanliness: H = (H + H^T) / 2
    let ht = transpose(&h);
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    let mut h_sym = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            h_sym[[i, j]] = (h[[i, j]] + ht[[i, j]]) / two;
        }
    }

    Ok((u, h_sym))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_polar_identity() {
        // Polar of I = (I, I)
        let eye = Array2::<f64>::eye(3);
        let (u, h) = polar_decomposition(&eye.view(), None, None).expect("polar identity");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(u[[i, j]], expected, epsilon = 1e-8);
                assert_abs_diff_eq!(h[[i, j]], expected, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_polar_u_orthogonal() {
        // U^T U should be identity
        let a = array![[3.0_f64, 1.0], [0.0, 2.0]];
        let (u, _h) = polar_decomposition(&a.view(), None, None).expect("polar u orthogonal");
        let ut = transpose(&u);
        let utu = matmul_nn(&ut, &u).expect("u^T u");
        assert_abs_diff_eq!(utu[[0, 0]], 1.0, epsilon = 1e-7);
        assert_abs_diff_eq!(utu[[1, 1]], 1.0, epsilon = 1e-7);
        assert_abs_diff_eq!(utu[[0, 1]], 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(utu[[1, 0]], 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_polar_reconstruct() {
        // U * H should reproduce A
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let (u, h) = polar_decomposition(&a.view(), None, None).expect("polar reconstruct");
        let a_rec = matmul_nn(&u, &h).expect("a_rec matmul");
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(a_rec[[i, j]], a[[i, j]], epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn test_polar_h_symmetric() {
        // H should be symmetric
        let a = array![[3.0_f64, 2.0], [1.0, 4.0]];
        let (_u, h) = polar_decomposition(&a.view(), None, None).expect("polar h symmetric");
        assert_abs_diff_eq!(h[[0, 1]], h[[1, 0]], epsilon = 1e-8);
    }

    #[test]
    fn test_unitary_factor_newton_identity() {
        let eye = Array2::<f64>::eye(2);
        let u = unitary_factor_newton(&eye.view(), None, None).expect("newton identity");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(u[[i, j]], expected, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_polar_via_svd_reconstruct() {
        let a = array![[2.0_f64, 0.5], [0.1, 3.0]];
        let (u, h) = polar_via_svd(&a.view()).expect("polar_via_svd reconstruct");
        let a_rec = matmul_nn(&u, &h).expect("a_rec svd");
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(a_rec[[i, j]], a[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_polar_via_svd_u_orthogonal() {
        let a = array![[4.0_f64, 2.0], [1.0, 3.0]];
        let (u, _h) = polar_via_svd(&a.view()).expect("polar_via_svd u orthogonal");
        let ut = transpose(&u);
        let utu = matmul_nn(&ut, &u).expect("utu svd");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(utu[[i, j]], expected, epsilon = 1e-8);
            }
        }
    }
}
