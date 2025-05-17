//! SciPy-compatible API wrappers for scirs2-linalg
//!
//! This module provides function signatures that match SciPy's linalg module
//! to ease migration and compatibility. The functions delegate to the main
//! scirs2-linalg implementations while providing the same parameter interface
//! as SciPy.

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::{basic, decomposition, eigen, solve};

/// Type alias for SVD decomposition result
pub type SvdResult<F> = (Option<Array2<F>>, Array1<F>, Option<Array2<F>>);

/// Compute the determinant of a matrix (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `overwrite_a` - Allow overwriting data in `a` (currently ignored)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
///
/// # Returns
/// * Determinant of the matrix
pub fn det<F>(a: &ArrayView2<F>, _overwrite_a: bool, check_finite: bool) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum,
{
    if check_finite {
        for &elem in a.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Matrix contains non-finite values".to_string(),
                ));
            }
        }
    }
    // Note: _overwrite_a is ignored in our implementation
    basic::det(a)
}

/// Compute the inverse of a matrix (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `overwrite_a` - Allow overwriting data in `a` (currently ignored)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
///
/// # Returns
/// * Inverse of the matrix
pub fn inv<F>(a: &ArrayView2<F>, _overwrite_a: bool, check_finite: bool) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum,
{
    if check_finite {
        for &elem in a.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Matrix contains non-finite values".to_string(),
                ));
            }
        }
    }
    // Note: _overwrite_a is ignored in our implementation
    basic::inv(a)
}

/// Compute eigenvalues and eigenvectors (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `b` - Optional matrix for generalized eigenvalue problem (not yet supported)
/// * `left` - Whether to compute left eigenvectors (not yet supported)
/// * `right` - Whether to compute right eigenvectors
/// * `overwrite_a` - Allow overwriting data in `a` (currently ignored)
/// * `overwrite_b` - Allow overwriting data in `b` (currently ignored)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
/// * `homogeneous_eigvals` - Whether to return homogeneous coordinates (not yet supported)
///
/// # Returns
/// * Tuple of (eigenvalues, eigenvectors) if right=true
#[allow(clippy::too_many_arguments)]
pub fn eig<F>(
    a: &ArrayView2<F>,
    b: Option<&ArrayView2<F>>,
    left: bool,
    _right: bool,
    _overwrite_a: bool,
    _overwrite_b: bool,
    check_finite: bool,
    homogeneous_eigvals: bool,
) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + 'static,
{
    if check_finite {
        for &elem in a.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Matrix contains non-finite values".to_string(),
                ));
            }
        }
        if let Some(b_mat) = b {
            for &elem in b_mat.iter() {
                if !elem.is_finite() {
                    return Err(LinalgError::ValueError(
                        "Matrix contains non-finite values".to_string(),
                    ));
                }
            }
        }
    }

    if b.is_some() {
        return Err(LinalgError::NotImplementedError(
            "Generalized eigenvalue problems not yet supported".to_string(),
        ));
    }

    if left {
        return Err(LinalgError::NotImplementedError(
            "Left eigenvectors not yet supported".to_string(),
        ));
    }

    if homogeneous_eigvals {
        return Err(LinalgError::NotImplementedError(
            "Homogeneous eigenvalues not yet supported".to_string(),
        ));
    }

    // Note: _overwrite_a, _overwrite_b are ignored in our implementation
    // Our implementation returns complex eigenvalues/eigenvectors
    // We return an error for now as SciPy compat would need real conversion handling
    Err(LinalgError::NotImplementedError(
        "Real eig conversion not yet implemented - use complex eig from main module".to_string(),
    ))
}

/// Compute eigenvalues and eigenvectors of symmetric/Hermitian matrix (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `b` - Optional matrix for generalized eigenvalue problem (not yet supported)
/// * `lower` - Whether to use lower triangular part
/// * `eigvals_only` - Whether to compute only eigenvalues
/// * `overwrite_a` - Allow overwriting data in `a` (currently ignored)
/// * `overwrite_b` - Allow overwriting data in `b` (currently ignored)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
/// * `subset_by_index` - Select eigenvalues by index range (not yet supported)
/// * `subset_by_value` - Select eigenvalues by value range (not yet supported)
/// * `driver` - LAPACK routine to use (currently ignored)
/// * `type_param` - Problem type for generalized eigenvalue problem (not yet supported)
///
/// # Returns
/// * Eigenvalues if eigvals_only=true, otherwise (eigenvalues, eigenvectors)
#[allow(clippy::too_many_arguments)]
pub fn eigh<F>(
    a: &ArrayView2<F>,
    b: Option<&ArrayView2<F>>,
    _lower: bool,
    eigvals_only: bool,
    _overwrite_a: bool,
    _overwrite_b: bool,
    check_finite: bool,
    subset_by_index: Option<(usize, usize)>,
    subset_by_value: Option<(F, F)>,
    _driver: Option<&str>,
    _type_param: u32,
) -> LinalgResult<(Array1<F>, Option<Array2<F>>)>
where
    F: Float + NumAssign + Sum + 'static,
{
    if check_finite {
        for &elem in a.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Matrix contains non-finite values".to_string(),
                ));
            }
        }
        if let Some(b_mat) = b {
            for &elem in b_mat.iter() {
                if !elem.is_finite() {
                    return Err(LinalgError::ValueError(
                        "Matrix contains non-finite values".to_string(),
                    ));
                }
            }
        }
    }

    if b.is_some() {
        return Err(LinalgError::NotImplementedError(
            "Generalized eigenvalue problems not yet supported".to_string(),
        ));
    }

    if subset_by_index.is_some() || subset_by_value.is_some() {
        return Err(LinalgError::NotImplementedError(
            "Eigenvalue subset selection not yet supported".to_string(),
        ));
    }

    // Note: lower, overwrite_a, overwrite_b, driver, type_param are not used
    if eigvals_only {
        let eigenvalues = eigen::eigvalsh(a)?;
        Ok((eigenvalues, None))
    } else {
        let (eigenvalues, eigenvectors) = eigen::eigh(a)?;
        Ok((eigenvalues, Some(eigenvectors)))
    }
}

/// LU decomposition (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `permute_l` - If true, return (L, U) with permuted L
/// * `overwrite_a` - Allow overwriting data in `a` (currently ignored)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
/// * `p_indices` - If true, return permutation as indices rather than matrix
///
/// # Returns
/// * Tuple of (P, L, U) or variations based on parameters
pub fn lu<F>(
    a: &ArrayView2<F>,
    permute_l: bool,
    _overwrite_a: bool,
    check_finite: bool,
    p_indices: bool,
) -> LinalgResult<(Array2<F>, Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum,
{
    if check_finite {
        for &elem in a.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Matrix contains non-finite values".to_string(),
                ));
            }
        }
    }

    if permute_l || p_indices {
        return Err(LinalgError::NotImplementedError(
            "Alternative LU output formats not yet supported".to_string(),
        ));
    }

    // Note: _overwrite_a is ignored in our implementation
    decomposition::lu(a)
}

/// QR decomposition (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `overwrite_a` - Allow overwriting data in `a` (currently ignored)
/// * `lwork` - Work array size (currently ignored)
/// * `mode` - Decomposition mode: 'full', 'r', 'economic', or 'raw'
/// * `pivoting` - Whether to use column pivoting (not yet supported)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
///
/// # Returns
/// * Tuple of (Q, R) or just R based on mode
pub fn qr<F>(
    a: &ArrayView2<F>,
    _overwrite_a: bool,
    _lwork: Option<usize>,
    mode: &str,
    pivoting: bool,
    check_finite: bool,
) -> LinalgResult<(Option<Array2<F>>, Array2<F>)>
where
    F: Float + NumAssign + Sum,
{
    if check_finite {
        for &elem in a.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Matrix contains non-finite values".to_string(),
                ));
            }
        }
    }

    if pivoting {
        return Err(LinalgError::NotImplementedError(
            "QR with pivoting not yet supported".to_string(),
        ));
    }

    // Note: overwrite_a and lwork are ignored in our implementation
    match mode {
        "full" | "economic" => {
            let (q, r) = decomposition::qr(a)?;
            Ok((Some(q), r))
        }
        "r" => {
            let (_, r) = decomposition::qr(a)?;
            Ok((None, r))
        }
        "raw" => Err(LinalgError::NotImplementedError(
            "Raw QR mode not yet supported".to_string(),
        )),
        _ => Err(LinalgError::InvalidInputError(format!(
            "Invalid QR mode: {}",
            mode
        ))),
    }
}

/// SVD decomposition (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `full_matrices` - If true, compute full-sized U and Vt matrices
/// * `compute_uv` - If true, compute U and Vt in addition to singular values
/// * `overwrite_a` - Allow overwriting data in `a` (currently ignored)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
/// * `lapack_driver` - LAPACK routine to use (currently ignored)
///
/// # Returns
/// * Tuple of (U, s, Vt) if compute_uv=true, otherwise just s
pub fn svd<F>(
    a: &ArrayView2<F>,
    full_matrices: bool,
    compute_uv: bool,
    _overwrite_a: bool,
    check_finite: bool,
    _lapack_driver: &str,
) -> LinalgResult<SvdResult<F>>
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand,
{
    if check_finite {
        for &elem in a.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Matrix contains non-finite values".to_string(),
                ));
            }
        }
    }

    // Note: overwrite_a and lapack_driver are ignored in our implementation
    if compute_uv {
        let (u, s, vt) = decomposition::svd(a, !full_matrices)?;
        Ok((Some(u), s, Some(vt)))
    } else {
        Err(LinalgError::NotImplementedError(
            "SVD without U,V computation not yet supported".to_string(),
        ))
    }
}

/// Cholesky decomposition (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `lower` - Whether to return lower triangular (true) or upper triangular (false)
/// * `overwrite_a` - Allow overwriting data in `a` (currently ignored)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
///
/// # Returns
/// * Lower or upper triangular Cholesky factor
pub fn cholesky<F>(
    a: &ArrayView2<F>,
    lower: bool,
    _overwrite_a: bool,
    check_finite: bool,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum,
{
    if check_finite {
        for &elem in a.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Matrix contains non-finite values".to_string(),
                ));
            }
        }
    }

    // Note: _overwrite_a is ignored in our implementation
    let l = decomposition::cholesky(a)?;

    if lower {
        Ok(l)
    } else {
        // Transpose to get upper triangular
        Ok(l.t().to_owned())
    }
}

/// Solve a linear system (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Coefficient matrix
/// * `b` - Right-hand side
/// * `lower` - Assume a is lower triangular
/// * `overwrite_a` - Allow overwriting data in `a` (currently ignored)
/// * `overwrite_b` - Allow overwriting data in `b` (currently ignored)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
/// * `assume_a` - Assumptions about matrix a ('gen', 'sym', 'her', 'pos')
/// * `transposed` - Whether to solve A.T x = b instead of A x = b
///
/// # Returns
/// * Solution x
#[allow(clippy::too_many_arguments)]
pub fn compat_solve<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    lower: bool,
    _overwrite_a: bool,
    _overwrite_b: bool,
    check_finite: bool,
    assume_a: Option<&str>,
    transposed: bool,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum,
{
    if check_finite {
        for &elem in a.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Matrix contains non-finite values".to_string(),
                ));
            }
        }
        for &elem in b.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Matrix contains non-finite values".to_string(),
                ));
            }
        }
    }

    if lower || assume_a.is_some() || transposed {
        return Err(LinalgError::NotImplementedError(
            "Specialized solve options not yet supported".to_string(),
        ));
    }

    // Note: _overwrite_a, _overwrite_b are ignored in our implementation
    solve::solve_multiple(a, b)
}

// Eigenvalue functions that match SciPy naming
pub use crate::eigen::eigvals;
pub use crate::eigen::eigvalsh;
pub use crate::eigen_specialized::banded_eigh as eig_banded;
pub use crate::eigen_specialized::banded_eigvalsh as eigvals_banded;
pub use crate::eigen_specialized::tridiagonal_eigh as eigh_tridiagonal;
pub use crate::eigen_specialized::tridiagonal_eigvalsh as eigvalsh_tridiagonal;

// Matrix functions
pub use crate::basic::matrix_power as fractional_matrix_power;
pub use crate::matrix_functions::expm;
pub use crate::matrix_functions::logm;
pub use crate::matrix_functions::sqrtm;
