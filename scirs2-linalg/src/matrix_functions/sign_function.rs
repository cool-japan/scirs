//! Matrix sign function for control theory and invariant subspace computation
//!
//! The matrix sign function sign(A) maps eigenvalues with positive real part to +1
//! and eigenvalues with negative real part to -1. It is used in:
//!
//! - Riccati equation solvers (invariant subspace extraction)
//! - Spectral projectors in control theory
//! - Polar decomposition via sign function
//!
//! # Algorithms
//!
//! - **Newton iteration**: S_{k+1} = (S_k + S_k^{-1}) / 2, quadratic convergence
//! - **Newton-Schulz iteration**: S_{k+1} = S_k(3I - S_k^2)/2, inversion-free,
//!   convergence when ||I - S_k^T S_k||_F < 1
//! - **Padé-based sign function**: Higher-order rational iterations
//!
//! # References
//!
//! - Roberts, J.D. (1980). "Linear model reduction and solution of the algebraic
//!   Riccati equation by use of the sign function."
//! - Kenney, C. & Laub, A.J. (1991). "Rational iterative methods for the matrix sign
//!   function." SIAM J. Numer. Anal.
//! - Byers, R. (1987). "Solving the algebraic Riccati equation with the matrix sign
//!   function." Linear Algebra Appl.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, One};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Floating-point trait alias for sign matrix functions.
pub trait SignFloat: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}
impl<F> SignFloat for F where F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn matmul_nn<F: SignFloat>(a: &Array2<F>, b: &Array2<F>) -> LinalgResult<Array2<F>> {
    let (m, k) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    if k != k2 {
        return Err(LinalgError::ShapeError(format!(
            "signm matmul: inner dims {} vs {}",
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

fn frobenius_norm<F: SignFloat>(a: &Array2<F>) -> F {
    let mut acc = F::zero();
    for &v in a.iter() {
        acc += v * v;
    }
    acc.sqrt()
}

/// LU factorization with partial pivoting.
fn lu_factorize<F: SignFloat>(a: &Array2<F>) -> LinalgResult<(Array2<F>, Vec<usize>)> {
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
                "Near-singular matrix in signm LU".into(),
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
            let lkk = lu[[k, k]];
            lu[[i, k]] /= lkk;
            for j in (k + 1)..n {
                let l_ik = lu[[i, k]];
                let u_kj = lu[[k, j]];
                lu[[i, j]] -= l_ik * u_kj;
            }
        }
    }
    Ok((lu, perm))
}

fn lu_solve<F: SignFloat>(lu: &Array2<F>, perm: &[usize], b: &Array2<F>) -> Array2<F> {
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

fn mat_inv<F: SignFloat>(a: &Array2<F>) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    let eye = Array2::<F>::eye(n);
    let (lu, perm) = lu_factorize(a)?;
    Ok(lu_solve(&lu, &perm, &eye))
}

// ---------------------------------------------------------------------------
// Newton iteration for sign function
// ---------------------------------------------------------------------------

/// Compute the matrix sign function via Newton iteration.
///
/// The Newton iteration:
/// ```text
/// S_{k+1} = (S_k + S_k^{-1}) / 2
/// ```
/// starting from S_0 = A converges quadratically to sign(A) when no eigenvalue
/// of A lies on the imaginary axis.
///
/// An optional determinant-based scaling factor mu_k can accelerate convergence:
/// ```text
/// mu_k = |det(S_k)|^{-1/n}
/// S_{k+1} = (mu_k S_k + mu_k^{-1} S_k^{-1}) / 2
/// ```
///
/// # Arguments
///
/// * `a` - Input square matrix (no purely imaginary eigenvalues)
/// * `max_iter` - Maximum iterations (default 100)
/// * `tol` - Convergence tolerance on ||S_{k+1} - S_k||_F / ||S_k||_F (default 1e-10)
///
/// # Returns
///
/// * sign(A) — matrix with eigenvalues ±1
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::sign_function::signm;
///
/// let a = array![[3.0_f64, 0.0], [0.0, -2.0]];
/// let s = signm(&a.view(), None, None).expect("signm failed");
/// assert!((s[[0, 0]] - 1.0).abs() < 1e-8);
/// assert!((s[[1, 1]] + 1.0).abs() < 1e-8);
/// ```
pub fn signm<F: SignFloat>(
    a: &ArrayView2<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "signm: matrix must be square".into(),
        ));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    let max_iter = max_iter.unwrap_or(100);
    let tol = tol.unwrap_or_else(|| F::from(1e-10).unwrap_or(F::epsilon()));
    let two = F::from(2.0).unwrap_or(F::one() + F::one());

    let mut s = a.to_owned();

    for _ in 0..max_iter {
        let s_inv = mat_inv(&s)?;

        // Determinant-based scaling for faster convergence
        let mu = compute_det_scaling(&s, n);

        let mut s_new = Array2::<F>::zeros((n, n));
        if (mu - F::one()).abs() < F::from(1e-6).unwrap_or(F::epsilon()) {
            // No scaling needed
            for i in 0..n {
                for j in 0..n {
                    s_new[[i, j]] = (s[[i, j]] + s_inv[[i, j]]) / two;
                }
            }
        } else {
            let mu_inv = F::one() / mu;
            for i in 0..n {
                for j in 0..n {
                    s_new[[i, j]] = (mu * s[[i, j]] + mu_inv * s_inv[[i, j]]) / two;
                }
            }
        }

        // Check convergence
        let mut diff = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                diff[[i, j]] = s_new[[i, j]] - s[[i, j]];
            }
        }
        let norm_s = frobenius_norm(&s_new);
        let rel_change = frobenius_norm(&diff) / (norm_s + F::from(1e-30).unwrap_or(F::epsilon()));

        s = s_new;

        if rel_change < tol {
            return Ok(s);
        }
    }

    Ok(s)
}

/// Compute determinant-based scaling |det(S)|^{-1/n}.
fn compute_det_scaling<F: SignFloat>(s: &Array2<F>, n: usize) -> F {
    if let Ok((lu, _)) = lu_factorize(s) {
        let mut log_abs_det = F::zero();
        let mut sign_changes = 0i32;
        for i in 0..n {
            let d = lu[[i, i]];
            if d.abs() < F::from(1e-15).unwrap_or(F::epsilon()) {
                return F::one();
            }
            if d < F::zero() {
                sign_changes += 1;
                log_abs_det += (-d).ln();
            } else {
                log_abs_det += d.ln();
            }
        }
        if sign_changes % 2 != 0 {
            // Odd permutation: negative determinant — no real scaling
            return F::one();
        }
        let exponent = -log_abs_det / F::from(n).unwrap_or(F::one());
        let mu = exponent.exp();
        // Clamp to reasonable range
        mu.max(F::from(0.1).unwrap_or(F::one()))
            .min(F::from(10.0).unwrap_or(F::one()))
    } else {
        F::one()
    }
}

// ---------------------------------------------------------------------------
// Newton-Schulz (inversion-free) iteration
// ---------------------------------------------------------------------------

/// Compute the matrix sign function via Newton-Schulz iteration (no matrix inversion).
///
/// The Newton-Schulz iteration:
/// ```text
/// S_{k+1} = S_k (3I - S_k^2) / 2
/// ```
/// is equivalent to the Newton iteration but avoids matrix inversion.
/// It converges when ||I - S_k^T S_k||_F < 1.
///
/// Typically initialized by first normalizing: S_0 = A / ||A||_F.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `max_iter` - Maximum iterations (default 200)
/// * `tol` - Convergence tolerance (default 1e-10)
///
/// # Returns
///
/// * sign(A)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::sign_function::signm_schulz;
///
/// let a = array![[2.0_f64, 0.0], [0.0, -3.0]];
/// let s = signm_schulz(&a.view(), None, None).expect("signm_schulz failed");
/// assert!((s[[0, 0]] - 1.0).abs() < 1e-6);
/// assert!((s[[1, 1]] + 1.0).abs() < 1e-6);
/// ```
pub fn signm_schulz<F: SignFloat>(
    a: &ArrayView2<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "signm_schulz: matrix must be square".into(),
        ));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    let max_iter = max_iter.unwrap_or(200);
    let tol = tol.unwrap_or_else(|| F::from(1e-10).unwrap_or(F::epsilon()));

    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    let three = F::from(3.0).unwrap_or(F::one() + F::one() + F::one());

    // Normalize to improve convergence: S = A / ||A||_F
    let norm_a = frobenius_norm(&a.to_owned());
    if norm_a < F::from(1e-15).unwrap_or(F::epsilon()) {
        return Err(LinalgError::SingularMatrixError(
            "signm_schulz: zero matrix has no sign function".into(),
        ));
    }

    let mut s = a.to_owned();
    for v in s.iter_mut() {
        *v /= norm_a;
    }

    for _ in 0..max_iter {
        let s2 = matmul_nn(&s, &s)?;

        // s_new = s * (3I - s^2) / 2
        let mut s_new = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let diag_term = if i == j { three } else { F::zero() };
                s_new[[i, j]] = (diag_term - s2[[i, j]]) / two;
            }
        }
        s_new = matmul_nn(&s, &s_new)?;

        // Check convergence
        let mut diff = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                diff[[i, j]] = s_new[[i, j]] - s[[i, j]];
            }
        }
        let rel_change = frobenius_norm(&diff)
            / (frobenius_norm(&s_new) + F::from(1e-30).unwrap_or(F::epsilon()));

        s = s_new;

        if rel_change < tol {
            return Ok(s);
        }
    }

    Ok(s)
}

// ---------------------------------------------------------------------------
// Spectral projectors via sign function
// ---------------------------------------------------------------------------

/// Compute spectral projectors for eigenvalue splitting.
///
/// Given A, compute the projectors onto the invariant subspaces corresponding
/// to eigenvalues with positive and negative real parts:
///
/// ```text
/// P_+ = (I + sign(A)) / 2   (projector onto Re(λ) > 0 subspace)
/// P_- = (I - sign(A)) / 2   (projector onto Re(λ) < 0 subspace)
/// ```
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `max_iter` - Maximum iterations for Newton iteration
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * `(P_plus, P_minus)` — spectral projectors
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::sign_function::spectral_projectors;
///
/// let a = array![[3.0_f64, 0.0], [0.0, -2.0]];
/// let (pp, pm) = spectral_projectors(&a.view(), None, None).expect("projectors failed");
/// // P_+ should project to positive eigenvalue subspace
/// assert!((pp[[0, 0]] - 1.0).abs() < 1e-6);
/// assert!((pm[[1, 1]] - 1.0).abs() < 1e-6);
/// ```
pub fn spectral_projectors<F: SignFloat>(
    a: &ArrayView2<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<(Array2<F>, Array2<F>)> {
    let n = a.nrows();
    let s = signm(a, max_iter, tol)?;
    let two = F::from(2.0).unwrap_or(F::one() + F::one());

    let mut p_plus = Array2::<F>::zeros((n, n));
    let mut p_minus = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let diag_term = if i == j { F::one() } else { F::zero() };
            p_plus[[i, j]] = (diag_term + s[[i, j]]) / two;
            p_minus[[i, j]] = (diag_term - s[[i, j]]) / two;
        }
    }

    Ok((p_plus, p_minus))
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
    fn test_signm_diagonal() {
        let a = array![[3.0_f64, 0.0], [0.0, -2.0]];
        let s = signm(&a.view(), None, None).expect("signm diagonal");
        assert_abs_diff_eq!(s[[0, 0]], 1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(s[[1, 1]], -1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(s[[0, 1]], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(s[[1, 0]], 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_signm_idempotent() {
        // sign(A)^2 = I
        let a = array![[2.0_f64, 1.0], [0.0, 3.0]];
        let s = signm(&a.view(), None, None).expect("signm idempotent");
        let s2 = matmul_nn(&s, &s).expect("s2 matmul");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(s2[[i, j]], expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_signm_all_positive() {
        // Positive definite diagonal: all signs are +1
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let s = signm(&a.view(), None, None).expect("signm all positive");
        assert_abs_diff_eq!(s[[0, 0]], 1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(s[[1, 1]], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_signm_schulz_diagonal() {
        let a = array![[2.0_f64, 0.0], [0.0, -3.0]];
        let s = signm_schulz(&a.view(), None, None).expect("signm_schulz diagonal");
        assert_abs_diff_eq!(s[[0, 0]], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(s[[1, 1]], -1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_spectral_projectors_diagonal() {
        let a = array![[3.0_f64, 0.0], [0.0, -2.0]];
        let (pp, pm) = spectral_projectors(&a.view(), None, None).expect("projectors diagonal");
        // P_+: projects to eigenvalue 3 (positive)
        assert_abs_diff_eq!(pp[[0, 0]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(pp[[1, 1]], 0.0, epsilon = 1e-6);
        // P_-: projects to eigenvalue -2 (negative)
        assert_abs_diff_eq!(pm[[0, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(pm[[1, 1]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_spectral_projectors_sum_to_identity() {
        // P_+ + P_- = I
        let a = array![[1.0_f64, 2.0], [0.5, 3.0]];
        let (pp, pm) = spectral_projectors(&a.view(), None, None).expect("projectors sum");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(pp[[i, j]] + pm[[i, j]], expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_signm_identity() {
        // sign(I) = I
        let eye = Array2::<f64>::eye(3);
        let s = signm(&eye.view(), None, None).expect("signm identity");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(s[[i, j]], expected, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_signm_commutes_with_inv() {
        // sign(A)^{-1} = sign(A) (since sign(A)^2 = I)
        let a = array![[2.0_f64, 1.0], [0.0, -3.0]];
        let s = signm(&a.view(), None, None).expect("signm commutes");
        let s_inv = mat_inv(&s).expect("s_inv");
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(s_inv[[i, j]], s[[i, j]], epsilon = 1e-6);
            }
        }
    }
}
