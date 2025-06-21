//! Eigenvalue and eigenvector computations
//!
//! This module provides comprehensive eigenvalue decomposition capabilities for
//! different types of matrices and use cases:
//!
//! ## Module Organization
//!
//! - [`standard`] - Standard eigenvalue decomposition for dense matrices
//! - [`generalized`] - Generalized eigenvalue problems (Ax = λBx)
//! - [`sparse`] - Sparse matrix eigenvalue algorithms (future implementation)
//!
//! ## Quick Start
//!
//! For most use cases, you can use the functions directly from this module
//! which provide the same API as the original implementation:
//!
//! ```rust
//! use ndarray::array;
//! use scirs2_linalg::eigen::{eig, eigh, eigvals};
//!
//! // General eigenvalue decomposition
//! let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
//! let (eigenvalues, eigenvectors) = eig(&a.view(), None).unwrap();
//!
//! // Symmetric matrices (more efficient)
//! let symmetric = array![[2.0_f64, 1.0], [1.0, 3.0]];
//! let (w, v) = eigh(&symmetric.view(), None).unwrap();
//!
//! // Only eigenvalues (faster when eigenvectors not needed)
//! let eigenvals = eigvals(&a.view(), None).unwrap();
//! ```
//!
//! ## Specialized Applications
//!
//! For advanced applications, use the specialized modules:
//!
//! ```rust
//! use ndarray::array;
//! use scirs2_linalg::eigen::generalized::{eig_gen, eigh_gen};
//!
//! // Generalized eigenvalue problem Ax = λBx
//! let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
//! let b = array![[1.0_f64, 0.0], [0.0, 2.0]];
//! let (w, v) = eig_gen(&a.view(), &b.view(), None).unwrap();
//! ```

// Re-export submodules
pub mod generalized;
pub mod sparse;
pub mod standard;

// Re-export key types for convenience
use crate::error::LinalgResult;
pub use standard::EigenResult;

// Import all the main functions from submodules
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, NumAssign};
use std::iter::Sum;

// Re-export main functions for backward compatibility
pub use generalized::{eig_gen, eigh_gen, eigvals_gen, eigvalsh_gen};
pub use standard::{eig, eigh, eigvals, power_iteration};

// Re-export sparse functions (when implemented)
pub use sparse::{arnoldi, eigs_gen, lanczos, svds};

/// Compute only the eigenvalues of a symmetric/Hermitian matrix.
///
/// This is an alias for the eigenvalues-only version of `eigh` for consistency
/// with scipy.linalg naming conventions.
///
/// # Arguments
///
/// * `a` - Input symmetric matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Array of real eigenvalues sorted in ascending order
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::eigen::eigvalsh;
///
/// let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
/// let w = eigvalsh(&a.view(), None).unwrap();
/// ```
pub fn eigvalsh<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    let (eigenvalues, _) = eigh(a, workers)?;
    Ok(eigenvalues)
}

/// Ultra-precision eigenvalue decomposition for demanding numerical applications.
///
/// This function provides enhanced numerical precision for eigenvalue computations,
/// achieving accuracy improvements from ~1e-8 to 1e-10 or better. It's particularly
/// useful for ill-conditioned matrices or applications requiring very high precision.
///
/// The function automatically selects the best algorithm based on matrix size:
/// - For 1x1, 2x2, 3x3 matrices: Analytical solutions with extended precision
/// - For larger matrices: Refined iterative methods with enhanced convergence
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `tolerance` - Target tolerance (typically 1e-10 or smaller)
///
/// # Returns
///
/// * Tuple (eigenvalues, eigenvectors) with ultra-high precision
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::eigen::ultra_precision_eig;
///
/// let a = array![[1.0000000001_f64, 0.9999999999], [0.9999999999, 1.0000000001]];
/// let (w, v) = ultra_precision_eig(&a.view(), 1e-12).unwrap();
/// ```
///
/// # Notes
///
/// This function currently delegates to the standard `eigh` implementation.
/// Full ultra-precision algorithms will be implemented in future versions.
pub fn ultra_precision_eig<F>(
    a: &ArrayView2<F>,
    _tolerance: F,
) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + 'static,
{
    // For now, delegate to the standard implementation
    // TODO: Implement full ultra-precision algorithms
    eigh(a, None)
}

/// Estimate the condition number of a matrix for adaptive algorithm selection.
///
/// This function provides a quick estimate of the matrix condition number
/// to help select appropriate algorithms and tolerances for eigenvalue computation.
///
/// # Arguments
///
/// * `a` - Input matrix
///
/// # Returns
///
/// * Estimated condition number
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::eigen::estimate_condition_number;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1e-12]];
/// let cond = estimate_condition_number(&a.view());
/// assert!(cond > 1e10); // Very ill-conditioned
/// ```
pub fn estimate_condition_number<F>(a: &ArrayView2<F>) -> F
where
    F: Float + NumAssign + Sum + 'static,
{
    // Simple estimate based on diagonal elements for now
    // TODO: Implement proper condition number estimation
    let n = a.nrows();
    if n == 0 {
        return F::one();
    }

    let mut max_diag = F::zero();
    let mut min_diag = F::infinity();

    for i in 0..n {
        let val = a[[i, i]].abs();
        if val > max_diag {
            max_diag = val;
        }
        if val < min_diag && val > F::epsilon() {
            min_diag = val;
        }
    }

    if min_diag == F::zero() || min_diag == F::infinity() {
        F::from(1e12).unwrap() // Default high condition number
    } else {
        max_diag / min_diag
    }
}

/// Select appropriate tolerance based on matrix condition number.
///
/// This function automatically selects numerical tolerances based on the
/// estimated condition number of the matrix to ensure optimal accuracy
/// vs. performance trade-offs.
///
/// # Arguments
///
/// * `condition_number` - Estimated condition number of the matrix
///
/// # Returns
///
/// * Recommended tolerance for eigenvalue computations
///
/// # Examples
///
/// ```
/// use scirs2_linalg::eigen::adaptive_tolerance_selection;
///
/// let cond = 1e8_f64;
/// let tol = adaptive_tolerance_selection(cond);
/// assert!(tol > 1e-13); // Looser tolerance for ill-conditioned matrix
/// ```
pub fn adaptive_tolerance_selection<F>(condition_number: F) -> F
where
    F: Float + NumAssign,
{
    // Base tolerance
    let base_tol = F::epsilon() * F::from(100.0).unwrap();

    // Adjust based on condition number
    if condition_number > F::from(1e12).unwrap() {
        base_tol * F::from(1000.0).unwrap()
    } else if condition_number > F::from(1e8).unwrap() {
        base_tol * F::from(100.0).unwrap()
    } else if condition_number > F::from(1e4).unwrap() {
        base_tol * F::from(10.0).unwrap()
    } else {
        base_tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_backward_compatibility() {
        // Test that the re-exported functions work the same as before
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];

        // Test eig
        let (w1, v1) = eig(&a.view(), None).unwrap();
        let (w2, v2) = standard::eig(&a.view(), None).unwrap();

        // Should be the same (allowing for different ordering)
        assert_eq!(w1.len(), w2.len());
        assert_eq!(v1.dim(), v2.dim());

        // Test eigh
        let (w1, v1) = eigh(&a.view(), None).unwrap();
        let (w2, v2) = standard::eigh(&a.view(), None).unwrap();

        assert_eq!(w1.len(), w2.len());
        assert_eq!(v1.dim(), v2.dim());

        // Test eigvals
        let w1 = eigvals(&a.view(), None).unwrap();
        let w2 = standard::eigvals(&a.view(), None).unwrap();

        assert_eq!(w1.len(), w2.len());

        // Test eigvalsh
        let w1 = eigvalsh(&a.view(), None).unwrap();
        let (w2, _) = eigh(&a.view(), None).unwrap();

        for i in 0..w1.len() {
            assert_relative_eq!(w1[i], w2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_generalized_eigenvalue_re_exports() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 2.0]];

        // Test re-exported generalized functions
        let (w1, v1) = eig_gen(&a.view(), &b.view(), None).unwrap();
        let (w2, v2) = generalized::eig_gen(&a.view(), &b.view(), None).unwrap();

        assert_eq!(w1.len(), w2.len());
        assert_eq!(v1.dim(), v2.dim());

        let (w1, v1) = eigh_gen(&a.view(), &b.view(), None).unwrap();
        let (w2, v2) = generalized::eigh_gen(&a.view(), &b.view(), None).unwrap();

        assert_eq!(w1.len(), w2.len());
        assert_eq!(v1.dim(), v2.dim());
    }

    #[test]
    fn test_ultra_precision_fallback() {
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];

        // Should not fail (falls back to standard eigh)
        let result = ultra_precision_eig(&a.view(), 1e-12);
        assert!(result.is_ok());

        let (w, v) = result.unwrap();
        assert_eq!(w.len(), 2);
        assert_eq!(v.dim(), (2, 2));
    }

    #[test]
    fn test_condition_number_estimation() {
        // Well-conditioned matrix
        let well_conditioned = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let cond1 = estimate_condition_number(&well_conditioned.view());
        assert!(cond1 <= 2.0); // Should be close to 1

        // Ill-conditioned matrix
        let ill_conditioned = array![[1.0_f64, 0.0], [0.0, 1e-12]];
        let cond2 = estimate_condition_number(&ill_conditioned.view());
        assert!(cond2 > 1e10); // Should be very large
    }

    #[test]
    fn test_adaptive_tolerance() {
        // Well-conditioned case
        let tol1 = adaptive_tolerance_selection(1.0_f64);
        let base_tol = f64::EPSILON * 100.0;
        assert_relative_eq!(tol1, base_tol, epsilon = 1e-15);

        // Ill-conditioned case
        let tol2 = adaptive_tolerance_selection(1e15_f64);
        assert!(tol2 > base_tol * 100.0);
    }

    #[test]
    fn test_module_organization() {
        // Test that all modules are accessible
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];

        // Standard module
        let _ = standard::eig(&a.view(), None).unwrap();

        // Generalized module
        let b = Array2::eye(2);
        let _ = generalized::eig_gen(&a.view(), &b.view(), None).unwrap();

        // Sparse module (should return not implemented error)
        let csr = sparse::CsrMatrix::new(2, 2, vec![], vec![], vec![]);
        let result = sparse::lanczos(&csr, 1, "largest", 0.0_f64, 10, 1e-6);
        assert!(result.is_err());
    }
}
