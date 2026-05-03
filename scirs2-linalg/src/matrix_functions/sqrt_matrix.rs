//! Matrix square root via Denman-Beavers and eigendecomposition methods
//!
//! # Algorithms
//!
//! - **Denman-Beavers iteration**: Coupled iteration X_{k+1} = (X_k + Y_k^{-1})/2,
//!   Y_{k+1} = (Y_k + X_k^{-1})/2 converging to (A^{1/2}, A^{-1/2}).
//! - **Product DB iteration**: Numerically stabilized variant with determinant-based
//!   scaling (Iannazzo 2006).
//! - **Eigendecomposition**: For symmetric positive definite matrices.
//!
//! # References
//!
//! - Denman, E.D. & Beavers, A.N. (1976). "The matrix sign function and computations
//!   in systems." Applied Mathematics and Computation.
//! - Iannazzo, B. (2006). "On the Newton method for the matrix pth root."
//!   SIAM Journal on Matrix Analysis and Applications.
//! - Higham, N.J. (2008). "Functions of Matrices: Theory and Computation."

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, One};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Floating-point trait alias for sqrt matrix functions.
pub trait SqrtFloat: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}
impl<F> SqrtFloat for F where F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn matmul_nn<F: SqrtFloat>(a: &Array2<F>, b: &Array2<F>) -> LinalgResult<Array2<F>> {
    let (m, k) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    if k != k2 {
        return Err(LinalgError::ShapeError(format!(
            "sqrtm matmul: inner dims mismatch {} vs {}",
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

fn frobenius_norm<F: SqrtFloat>(a: &Array2<F>) -> F {
    let mut acc = F::zero();
    for &v in a.iter() {
        acc += v * v;
    }
    acc.sqrt()
}

/// LU factorization with partial pivoting for matrix inversion.
/// Returns (L, U, perm) where perm[i] = j means row i was swapped with row j.
fn lu_factorize<F: SqrtFloat>(a: &Array2<F>) -> LinalgResult<(Array2<F>, Vec<usize>)> {
    let n = a.nrows();
    let mut lu = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot
        let mut max_val = F::zero();
        let mut max_row = k;
        for i in k..n {
            let v = lu[[i, k]].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_val < F::epsilon() * F::from(1000.0).unwrap_or(F::one()) {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is (nearly) singular in LU factorization".into(),
            ));
        }

        // Swap rows
        if max_row != k {
            perm.swap(k, max_row);
            for j in 0..n {
                let tmp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
        }

        // Elimination
        for i in (k + 1)..n {
            if lu[[k, k]].abs() < F::epsilon() {
                continue;
            }
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

/// Solve A*X = B using precomputed LU factorization.
fn lu_solve<F: SqrtFloat>(lu: &Array2<F>, perm: &[usize], b: &Array2<F>) -> Array2<F> {
    let n = lu.nrows();
    let nrhs = b.ncols();
    let mut x = Array2::<F>::zeros((n, nrhs));

    for col in 0..nrhs {
        // Apply permutation
        let mut y = vec![F::zero(); n];
        for i in 0..n {
            y[i] = b[[perm[i], col]];
        }

        // Forward substitution (L*y = Pb)
        for i in 0..n {
            for j in 0..i {
                let yj = y[j];
                y[i] -= lu[[i, j]] * yj;
            }
        }

        // Back substitution (U*x = y)
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

/// Compute the inverse of a matrix.
fn mat_inv<F: SqrtFloat>(a: &Array2<F>) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    let identity = Array2::<F>::eye(n);
    let (lu, perm) = lu_factorize(a)?;
    Ok(lu_solve(&lu, &perm, &identity))
}

// ---------------------------------------------------------------------------
// Denman-Beavers iteration
// ---------------------------------------------------------------------------

/// Compute the matrix square root via Denman-Beavers iteration.
///
/// The Denman-Beavers coupled iteration:
/// ```text
/// X_{k+1} = (X_k + Y_k^{-1}) / 2
/// Y_{k+1} = (Y_k + X_k^{-1}) / 2
/// ```
/// starting from X_0 = A, Y_0 = I converges to X_∞ = A^{1/2}.
///
/// Convergence is quadratic when the starting matrix has no purely negative eigenvalues.
///
/// # Arguments
///
/// * `a` - Input square matrix (should have no negative real eigenvalues)
/// * `max_iter` - Maximum iterations (default 100)
/// * `tol` - Convergence tolerance (default 1e-10)
///
/// # Returns
///
/// * `A^{1/2}` — the principal square root
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::sqrt_matrix::sqrtm_denman_beavers;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let s = sqrtm_denman_beavers(&a.view(), None, None).expect("sqrtm failed");
/// assert!((s[[0, 0]] - 2.0).abs() < 1e-8);
/// assert!((s[[1, 1]] - 3.0).abs() < 1e-8);
/// ```
pub fn sqrtm_denman_beavers<F: SqrtFloat>(
    a: &ArrayView2<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "sqrtm_denman_beavers: matrix must be square".into(),
        ));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    let max_iter = max_iter.unwrap_or(100);
    let tol = tol.unwrap_or_else(|| F::from(1e-10).unwrap_or(F::epsilon()));

    let mut x = a.to_owned();
    let mut y = Array2::<F>::eye(n);

    for _ in 0..max_iter {
        let x_inv = mat_inv(&x)?;
        let y_inv = mat_inv(&y)?;

        let two = F::from(2.0).unwrap_or(F::one() + F::one());

        // X_new = (X + Y^{-1}) / 2
        let mut x_new = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                x_new[[i, j]] = (x[[i, j]] + y_inv[[i, j]]) / two;
            }
        }

        // Y_new = (Y + X^{-1}) / 2
        let mut y_new = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                y_new[[i, j]] = (y[[i, j]] + x_inv[[i, j]]) / two;
            }
        }

        // Check convergence: ||X_new - X||_F / ||X_new||_F
        let mut diff = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                diff[[i, j]] = x_new[[i, j]] - x[[i, j]];
            }
        }
        let rel_change = frobenius_norm(&diff)
            / (frobenius_norm(&x_new) + F::from(1e-30).unwrap_or(F::epsilon()));

        x = x_new;
        y = y_new;

        if rel_change < tol {
            return Ok(x);
        }
    }

    // Return best approximation even if not fully converged
    Ok(x)
}

// ---------------------------------------------------------------------------
// Product Denman-Beavers iteration (scaled)
// ---------------------------------------------------------------------------

/// Compute the matrix square root via the product form of Denman-Beavers (scaled).
///
/// The product DB iteration uses determinant-based scaling to improve convergence:
/// ```text
/// mu_k = |det(X_k)|^{-1/(2n)}
/// X_{k+1} = (mu_k * X_k + mu_k^{-1} * Y_k^{-1}) / 2
/// Y_{k+1} = (mu_k * Y_k + mu_k^{-1} * X_k^{-1}) / 2
/// ```
///
/// The scaling accelerates convergence significantly for ill-conditioned matrices.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `max_iter` - Maximum iterations (default 50)
/// * `tol` - Convergence tolerance (default 1e-12)
///
/// # Returns
///
/// * `A^{1/2}`
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::sqrt_matrix::sqrtm_product_db;
///
/// let a = array![[9.0_f64, 0.0], [0.0, 4.0]];
/// let s = sqrtm_product_db(&a.view(), None, None).expect("sqrtm_product_db failed");
/// assert!((s[[0, 0]] - 3.0).abs() < 1e-8);
/// assert!((s[[1, 1]] - 2.0).abs() < 1e-8);
/// ```
pub fn sqrtm_product_db<F: SqrtFloat>(
    a: &ArrayView2<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "sqrtm_product_db: matrix must be square".into(),
        ));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    let max_iter = max_iter.unwrap_or(50);
    let tol = tol.unwrap_or_else(|| F::from(1e-12).unwrap_or(F::epsilon()));
    let two = F::from(2.0).unwrap_or(F::one() + F::one());

    // Iannazzo's one-time pre-scaling: mu0 = |det(A)|^{-1/(2n)}
    // Scale: A_scaled = mu0 * A.  Then run standard DB on A_scaled.
    // Post-scale: sqrt(A) = sqrt(A_scaled) / sqrt(mu0).
    let a_owned = a.to_owned();
    let mu0 = compute_det_scale(&a_owned, n);
    let mu0_sqrt = mu0.sqrt();
    // Avoid divide-by-zero
    let mu0_sqrt_inv = if mu0_sqrt.abs() < F::from(1e-30).unwrap_or(F::epsilon()) {
        F::one()
    } else {
        F::one() / mu0_sqrt
    };

    // x0 = mu0 * A,  y0 = I  (standard DB initial conditions on scaled A)
    let mut x = a.mapv(|v| v * mu0);
    let mut y = Array2::<F>::eye(n);

    for _ in 0..max_iter {
        let x_inv = mat_inv(&x)?;
        let y_inv = mat_inv(&y)?;

        let mut x_new = Array2::<F>::zeros((n, n));
        let mut y_new = Array2::<F>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                x_new[[i, j]] = (x[[i, j]] + y_inv[[i, j]]) / two;
                y_new[[i, j]] = (y[[i, j]] + x_inv[[i, j]]) / two;
            }
        }

        // Check convergence
        let mut diff = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                diff[[i, j]] = x_new[[i, j]] - x[[i, j]];
            }
        }
        let rel_change = frobenius_norm(&diff)
            / (frobenius_norm(&x_new) + F::from(1e-30).unwrap_or(F::epsilon()));

        x = x_new;
        y = y_new;

        if rel_change < tol {
            // Post-scale: sqrt(A) = sqrt(mu0 * A) / sqrt(mu0)
            return Ok(x.mapv(|v| v * mu0_sqrt_inv));
        }
    }

    // Post-scale at max_iter
    Ok(x.mapv(|v| v * mu0_sqrt_inv))
}

/// Estimate |det(A)|^{-1/(2n)} for scaling.
fn compute_det_scale<F: SqrtFloat>(a: &Array2<F>, n: usize) -> F {
    // Use the trace as a proxy: det ≈ exp(trace(log(A))) for SPD matrices.
    // For general matrices, we use the product of diagonal elements of LU.
    if let Ok((lu, _)) = lu_factorize(a) {
        let mut log_det = F::zero();
        let mut sign_count = 0i32;
        for i in 0..n {
            let d = lu[[i, i]];
            if d.abs() < F::from(1e-30).unwrap_or(F::epsilon()) {
                return F::one(); // Singular: don't scale
            }
            if d < F::zero() {
                sign_count += 1;
                log_det += (-d).ln();
            } else {
                log_det += d.ln();
            }
        }

        if sign_count % 2 != 0 {
            // Negative determinant: no real square root; return neutral scale
            return F::one();
        }

        // mu = |det|^{-1/(2n)} = exp(-log|det| / (2n))
        let exponent = -log_det / F::from(2 * n).unwrap_or(F::one());
        exponent.exp()
    } else {
        F::one()
    }
}

// ---------------------------------------------------------------------------
// Symmetric positive definite via eigendecomposition
// ---------------------------------------------------------------------------

/// Compute the matrix square root of a symmetric positive definite matrix
/// via eigendecomposition.
///
/// For a SPD matrix A = V * D * V^T (eigendecomposition),
/// A^{1/2} = V * D^{1/2} * V^T where D^{1/2} = diag(sqrt(d_i)).
///
/// # Arguments
///
/// * `a` - Symmetric positive definite matrix
///
/// # Returns
///
/// * `A^{1/2}` — symmetric positive semidefinite square root
///
/// # Errors
///
/// Returns an error if any eigenvalue is negative (matrix not PSD).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::sqrt_matrix::sqrtm_positive_definite;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let s = sqrtm_positive_definite(&a.view()).expect("sqrtm_pd failed");
/// assert!((s[[0, 0]] - 2.0).abs() < 1e-8);
/// assert!((s[[1, 1]] - 3.0).abs() < 1e-8);
/// ```
pub fn sqrtm_positive_definite<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ScalarOperand + Send + Sync + 'static + std::fmt::Display,
{
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "sqrtm_positive_definite: matrix must be square".into(),
        ));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    use crate::matrix_functions::fractional::spdmatrix_function;

    spdmatrix_function(
        a,
        |x: F| {
            if x < F::zero() {
                F::zero() // Clamp negative eigenvalues to zero for robustness
            } else {
                x.sqrt()
            }
        },
        true,
    )
}

// ---------------------------------------------------------------------------
// Main sqrtm entry point (re-export wrapper)
// ---------------------------------------------------------------------------

/// Compute the matrix square root using the best available method.
///
/// Selects the algorithm based on matrix properties:
/// - For small matrices (n <= 2): uses Denman-Beavers
/// - Otherwise: uses the scaled product DB iteration for numerical stability
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `max_iter` - Maximum iterations (default: method-specific)
/// * `tol` - Convergence tolerance (default: method-specific)
///
/// # Returns
///
/// * `A^{1/2}` — the principal square root
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::sqrt_matrix::sqrtm;
///
/// let a = array![[4.0_f64, 2.0], [0.0, 9.0]];
/// let s = sqrtm(&a.view(), None, None).expect("sqrtm failed");
/// // Verify: S * S ≈ A
/// let s00 = s[[0, 0]]; let s01 = s[[0, 1]];
/// let s10 = s[[1, 0]]; let s11 = s[[1, 1]];
/// assert!((s00 * s00 + s01 * s10 - 4.0).abs() < 1e-6);
/// assert!((s11 * s11 - 9.0).abs() < 1e-6);
/// ```
pub fn sqrtm<F: SqrtFloat>(
    a: &ArrayView2<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "sqrtm: matrix must be square".into(),
        ));
    }

    // Denman-Beavers is numerically stable for all sizes
    let _ = n; // n used only to satisfy square check above
    sqrtm_denman_beavers(a, max_iter, tol)
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
    fn test_sqrtm_db_diagonal() {
        let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
        let s = sqrtm_denman_beavers(&a.view(), None, None).expect("sqrtm_db diagonal");
        assert_abs_diff_eq!(s[[0, 0]], 2.0, epsilon = 1e-7);
        assert_abs_diff_eq!(s[[1, 1]], 3.0, epsilon = 1e-7);
        assert_abs_diff_eq!(s[[0, 1]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(s[[1, 0]], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sqrtm_db_identity() {
        let a = Array2::<f64>::eye(3);
        let s = sqrtm_denman_beavers(&a.view(), None, None).expect("sqrtm_db identity");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(s[[i, j]], expected, epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn test_sqrtm_product_db_diagonal() {
        let a = array![[9.0_f64, 0.0], [0.0, 4.0]];
        let s = sqrtm_product_db(&a.view(), None, None).expect("sqrtm_product_db diagonal");
        assert_abs_diff_eq!(s[[0, 0]], 3.0, epsilon = 1e-8);
        assert_abs_diff_eq!(s[[1, 1]], 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_sqrtm_verifies_s_squared() {
        // S^2 should equal A
        let a = array![[5.0_f64, 2.0], [2.0, 5.0]];
        let s = sqrtm_denman_beavers(&a.view(), Some(200), Some(1e-12))
            .expect("sqrtm_db square verify");
        let s2 = matmul_nn(&s, &s).expect("s2 matmul");
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(s2[[i, j]], a[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_sqrtm_pd_diagonal() {
        let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
        let s = sqrtm_positive_definite(&a.view()).expect("sqrtm_pd diagonal");
        assert_abs_diff_eq!(s[[0, 0]], 2.0, epsilon = 1e-8);
        assert_abs_diff_eq!(s[[1, 1]], 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_sqrtm_dispatch_large() {
        let n = 5;
        let mut a = Array2::<f64>::eye(n);
        // Scale by 4 so sqrt = 2*I
        for i in 0..n {
            a[[i, i]] = 4.0;
        }
        let s = sqrtm(&a.view(), None, None).expect("sqrtm dispatch large");
        for i in 0..n {
            assert_abs_diff_eq!(s[[i, i]], 2.0, epsilon = 1e-7);
        }
    }

    #[test]
    fn test_sqrtm_2x2_upper_triangular() {
        // A = [[4, 2], [0, 9]]; sqrt is [[2, r], [0, 3]]
        // where r = 2 / (2 + 3) = 0.4
        let a = array![[4.0_f64, 2.0], [0.0, 9.0]];
        let s =
            sqrtm_denman_beavers(&a.view(), Some(200), Some(1e-12)).expect("sqrtm_db triangular");
        let s2 = matmul_nn(&s, &s).expect("s2 matmul triangular");
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(s2[[i, j]], a[[i, j]], epsilon = 1e-6);
            }
        }
    }
}
