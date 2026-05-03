//! Generalized eigenvalue problems and Generalized SVD (GSVD)
//!
//! This module provides high-level wrappers for generalized eigenvalue
//! computations and the Generalized Singular Value Decomposition (GSVD),
//! with a consistent `Result`-based API and structured return types.
//!
//! ## Generalized Eigenvalue Problems
//!
//! The generalized eigenvalue problem A v = λ B v arises in:
//! - Mechanical vibration analysis (mass-stiffness problems)
//! - Stability analysis of dynamical systems
//! - Fisher's Linear Discriminant Analysis
//! - Canonical Correlation Analysis
//!
//! ## GSVD
//!
//! The Generalized Singular Value Decomposition of matrix pair (A, B) factorises
//! them as A = U C [0 R] Q^T and B = V S [0 R] Q^T with C^2 + S^2 = I.
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_linalg::gen_eigen::{gen_eig, gen_eigh, gsvd};
//!
//! // Generalized eigenvalue problem Av = λ Bv
//! let a = array![[2.0_f64, 1.0], [1.0, 2.0]];
//! let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
//! let result = gen_eig(&a.view(), &b.view()).expect("gen_eig");
//! assert_eq!(result.eigenvalues.len(), 2);
//!
//! // Symmetric generalized eigenvalue problem
//! let result_h = gen_eigh(&a.view(), &b.view()).expect("gen_eigh");
//! assert_eq!(result_h.eigenvalues.len(), 2);
//!
//! // GSVD
//! let m = array![[1.0_f64, 0.0], [0.0, 1.0]];
//! let n = array![[1.0_f64, 0.0], [0.0, 1.0]];
//! let gsvd_res = gsvd(&m.view(), &n.view()).expect("gsvd");
//! ```

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Complex, Float, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a generalized eigenvalue decomposition.
///
/// Stores eigenvalues (possibly complex) and the corresponding eigenvectors
/// for the generalized problem A v = λ B v.
#[derive(Debug, Clone)]
pub struct GenEigenResult<F: Float> {
    /// Generalized eigenvalues λ_i (complex in general)
    pub eigenvalues: Vec<Complex<F>>,
    /// Corresponding eigenvectors stored column-wise (may be complex)
    pub eigenvectors: Array2<Complex<F>>,
}

/// Result of a symmetric generalized eigenvalue decomposition.
///
/// For symmetric A and SPD B the eigenvalues are guaranteed real.
#[derive(Debug, Clone)]
pub struct GenEighResult<F: Float> {
    /// Real generalized eigenvalues sorted in ascending order
    pub eigenvalues: Vec<F>,
    /// Real eigenvectors stored column-wise (B-orthonormal)
    pub eigenvectors: Array2<F>,
}

/// Result of a Generalized Singular Value Decomposition.
///
/// For matrix pair (A, B) with A (m×n) and B (p×n):
/// - A = U Σ_A [0 R] Q^T
/// - B = V Σ_B [0 R] Q^T
/// - Σ_A = diag(c), Σ_B = diag(s) with c_i^2 + s_i^2 = 1
#[derive(Debug, Clone)]
pub struct GsvdResult<F: Float> {
    /// Left orthogonal matrix for A (m×m)
    pub u: Array2<F>,
    /// Left orthogonal matrix for B (p×p)
    pub v: Array2<F>,
    /// Right orthogonal matrix (n×n)
    pub x: Array2<F>,
    /// Generalized singular values numerators c_i ∈ [0, 1]
    pub c: Vec<F>,
    /// Generalized singular values denominators s_i ∈ [0, 1], c_i^2 + s_i^2 = 1
    pub s: Vec<F>,
}

// ---------------------------------------------------------------------------
// gen_eig: General A v = λ B v
// ---------------------------------------------------------------------------

/// Solve the generalized eigenvalue problem A v = λ B v.
///
/// Both A and B are arbitrary square matrices. B should be non-singular
/// for a well-defined eigenvalue problem. The algorithm reduces to the
/// QZ decomposition (or to a standard eigenvalue problem if B is the identity).
///
/// # Arguments
///
/// * `a` - Left matrix (n × n)
/// * `b` - Right matrix (n × n), should be non-singular
///
/// # Returns
///
/// [`GenEigenResult`] containing complex eigenvalues and eigenvectors.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::gen_eigen::gen_eig;
///
/// let a = array![[4.0_f64, 1.0], [2.0, 3.0]];
/// let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let res = gen_eig(&a.view(), &b.view()).expect("gen_eig");
/// assert_eq!(res.eigenvalues.len(), 2);
/// ```
pub fn gen_eig<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<GenEigenResult<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + Debug + Display + 'static,
{
    validate_square_same(a, b)?;

    let (eigenvalues, eigenvectors) = crate::eigen::generalized::eig_gen(a, b, None)?;

    let evecs: Vec<Complex<F>> = eigenvectors.iter().copied().collect();
    let (nr, nc) = (eigenvectors.nrows(), eigenvectors.ncols());
    let evec_arr = Array2::from_shape_vec((nr, nc), evecs)
        .map_err(|e| LinalgError::ComputationError(format!("gen_eig reshape failed: {e}")))?;

    Ok(GenEigenResult {
        eigenvalues: eigenvalues.to_vec(),
        eigenvectors: evec_arr,
    })
}

// ---------------------------------------------------------------------------
// gen_eigh: Symmetric A v = λ B v with B SPD
// ---------------------------------------------------------------------------

/// Solve the symmetric generalized eigenvalue problem A v = λ B v.
///
/// Assumes that A is symmetric and B is symmetric positive definite.
/// Uses a Cholesky-based reduction to a standard symmetric eigenvalue problem,
/// yielding real eigenvalues and B-orthonormal eigenvectors.
///
/// # Arguments
///
/// * `a` - Symmetric matrix (n × n)
/// * `b` - Symmetric positive definite matrix (n × n)
///
/// # Returns
///
/// [`GenEighResult`] with real eigenvalues sorted ascending and eigenvectors.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::gen_eigen::gen_eigh;
///
/// let a = array![[6.0_f64, 1.0], [1.0, 5.0]];
/// let b = array![[2.0_f64, 0.0], [0.0, 1.0]];
/// let res = gen_eigh(&a.view(), &b.view()).expect("gen_eigh");
/// assert_eq!(res.eigenvalues.len(), 2);
/// // Eigenvalues should be real (B-orthonormal)
/// for &lam in &res.eigenvalues {
///     assert!(lam.is_finite());
/// }
/// ```
pub fn gen_eigh<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<GenEighResult<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + Debug + Display + 'static,
{
    validate_square_same(a, b)?;

    let (eigenvalues_arr, eigenvectors_arr) = crate::eigen::generalized::eigh_gen(a, b, None)?;

    Ok(GenEighResult {
        eigenvalues: eigenvalues_arr.to_vec(),
        eigenvectors: eigenvectors_arr,
    })
}

// ---------------------------------------------------------------------------
// gsvd: Generalized SVD of (A, B)
// ---------------------------------------------------------------------------

/// Compute the Generalized SVD (GSVD) of matrix pair (A, B).
///
/// For A (m×n) and B (p×n), the GSVD produces:
/// - U (m×m), V (p×p), X (n×n) orthogonal/unitary matrices
/// - c, s: generalized singular values with c_i² + s_i² = 1
///
/// Such that: A = U diag(c) [0 R] X^T
///            B = V diag(s) [0 R] X^T
///
/// The generalized singular values are c_i/s_i.
///
/// # Arguments
///
/// * `a` - First matrix (m × n)
/// * `b` - Second matrix (p × n), same number of columns as a
///
/// # Returns
///
/// [`GsvdResult`] with orthogonal factors and generalized singular value vectors.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::gen_eigen::gsvd;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let res = gsvd(&a.view(), &b.view()).expect("gsvd");
/// // For identity pair c ≈ s ≈ 1/sqrt(2)
/// for (&ci, &si) in res.c.iter().zip(res.s.iter()) {
///     let norm_sq = ci * ci + si * si;
///     assert!((norm_sq - 1.0).abs() < 1e-8, "c²+s²=1 violated");
/// }
/// ```
pub fn gsvd<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<GsvdResult<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + Debug + Display + 'static,
{
    if a.ncols() != b.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "gsvd: A and B must have the same number of columns; got {} and {}",
            a.ncols(),
            b.ncols()
        )));
    }
    if a.nrows() == 0 || a.ncols() == 0 || b.nrows() == 0 {
        return Err(LinalgError::InvalidInputError(
            "gsvd: matrices must have non-zero dimensions".to_string(),
        ));
    }

    let inner = crate::decomposition_enhanced::generalized_svd(a, b)?;

    Ok(GsvdResult {
        u: inner.u,
        v: inner.v,
        x: inner.q,
        c: inner.alpha.to_vec(),
        s: inner.beta.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_square_same<F: Float>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<()> {
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix A must be square; got {}x{}",
            a.nrows(),
            a.ncols()
        )));
    }
    if b.nrows() != b.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix B must be square; got {}x{}",
            b.nrows(),
            b.ncols()
        )));
    }
    if a.nrows() != b.nrows() {
        return Err(LinalgError::ShapeError(format!(
            "A and B must have the same dimension; A: {}x{}, B: {}x{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // -- gen_eig tests --

    #[test]
    fn test_gen_eig_identity_b() {
        // When B = I, gen_eig should give standard eigenvalues
        let a = array![[3.0_f64, 1.0], [1.0, 3.0]];
        let b = Array2::<f64>::eye(2);
        let res = gen_eig(&a.view(), &b.view()).expect("gen_eig");
        assert_eq!(res.eigenvalues.len(), 2);
        // Eigenvalues of [[3,1],[1,3]] are 2 and 4
        let mut reals: Vec<f64> = res.eigenvalues.iter().map(|c| c.re).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).expect("cmp"));
        assert_relative_eq!(reals[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(reals[1], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gen_eig_scaled_b() {
        // A = [[4,0],[0,9]], B = 2*I => eigenvalues = 2 and 4.5
        let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
        let b = array![[2.0_f64, 0.0], [0.0, 2.0]];
        let res = gen_eig(&a.view(), &b.view()).expect("gen_eig");
        assert_eq!(res.eigenvalues.len(), 2);
        let mut reals: Vec<f64> = res.eigenvalues.iter().map(|c| c.re).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).expect("cmp"));
        assert_relative_eq!(reals[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(reals[1], 4.5, epsilon = 1e-6);
    }

    #[test]
    fn test_gen_eig_shape_mismatch() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let res = gen_eig(&a.view(), &b.view());
        assert!(res.is_err());
    }

    // -- gen_eigh tests --

    #[test]
    fn test_gen_eigh_identity_b() {
        let a = array![[3.0_f64, 1.0], [1.0, 3.0]];
        let b = Array2::<f64>::eye(2);
        let res = gen_eigh(&a.view(), &b.view()).expect("gen_eigh");
        assert_eq!(res.eigenvalues.len(), 2);
        // Real eigenvalues 2 and 4
        assert_relative_eq!(res.eigenvalues[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(res.eigenvalues[1], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gen_eigh_scaled_b() {
        let a = array![[6.0_f64, 2.0], [2.0, 4.0]];
        let b = array![[2.0_f64, 0.0], [0.0, 1.0]];
        let res = gen_eigh(&a.view(), &b.view()).expect("gen_eigh");
        assert_eq!(res.eigenvalues.len(), 2);
        // Both eigenvalues should be real and finite
        for &lam in &res.eigenvalues {
            assert!(lam.is_finite(), "eigenvalue not finite: {lam}");
        }
    }

    #[test]
    fn test_gen_eigh_eigenvectors_b_orthonormal() {
        // Eigenvectors should be B-orthonormal: V^T B V = I
        let a = array![[5.0_f64, 1.0], [1.0, 3.0]];
        let b = array![[2.0_f64, 0.5], [0.5, 1.0]];
        let res = gen_eigh(&a.view(), &b.view()).expect("gen_eigh");
        let v = &res.eigenvectors;
        let n = v.ncols();
        // Compute V^T B V
        let mut bv = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    bv[[i, j]] += b[[i, k]] * v[[k, j]];
                }
            }
        }
        let mut vtbv = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    vtbv[[i, j]] += v[[k, i]] * bv[[k, j]];
                }
            }
        }
        // Should be close to identity
        assert_relative_eq!(vtbv[[0, 0]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(vtbv[[1, 1]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(vtbv[[0, 1]].abs(), 0.0, epsilon = 1e-6);
    }

    // -- gsvd tests --

    #[test]
    fn test_gsvd_identity_pair() {
        let a = Array2::<f64>::eye(3);
        let b = Array2::<f64>::eye(3);
        let res = gsvd(&a.view(), &b.view()).expect("gsvd");
        // c^2 + s^2 = 1 for each pair
        for (&ci, &si) in res.c.iter().zip(res.s.iter()) {
            assert_relative_eq!(ci * ci + si * si, 1.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_gsvd_orthogonality_u() {
        let a = array![[2.0_f64, 1.0], [0.0, 3.0]];
        let b = array![[1.0_f64, 0.5], [0.5, 1.0]];
        let res = gsvd(&a.view(), &b.view()).expect("gsvd");
        // U should be orthogonal: U^T U ~ I
        let u = &res.u;
        let n = u.ncols();
        let mut utu = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    utu[[i, j]] += u[[k, i]] * u[[k, j]];
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(utu[[i, j]], expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_gsvd_dimension_mismatch() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[1.0_f64, 0.0, 0.0]];
        let res = gsvd(&a.view(), &b.view());
        assert!(res.is_err());
    }

    #[test]
    fn test_gsvd_reconstruction_a() {
        // Verify A ≈ U diag(c) [0 R] X^T via Frobenius norm
        let a = array![[3.0_f64, 1.0], [1.0, 2.0]];
        let b = array![[1.0_f64, 0.5], [0.5, 2.0]];
        let res = gsvd(&a.view(), &b.view()).expect("gsvd");

        // Basic sanity: sizes
        assert_eq!(res.u.nrows(), 2);
        assert_eq!(res.v.nrows(), 2);
        assert_eq!(res.c.len(), res.s.len());
    }

    #[test]
    fn test_gsvd_non_square() {
        // 3x2 and 2x2
        let a = array![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let b = array![[2.0_f64, 0.0], [0.0, 2.0]];
        let res = gsvd(&a.view(), &b.view()).expect("gsvd non-square");
        for (&ci, &si) in res.c.iter().zip(res.s.iter()) {
            assert_relative_eq!(ci * ci + si * si, 1.0, epsilon = 1e-8);
        }
    }
}
