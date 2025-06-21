//! SciPy-compatible API wrappers for scirs2-linalg
//!
//! This module provides function signatures that match SciPy's linalg module
//! to ease migration and compatibility. The functions delegate to the main
//! scirs2-linalg implementations while providing the same parameter interface
//! as SciPy.
//!
//! ## Supported Functions
//!
//! ### Basic Matrix Operations
//! - `det()` - Matrix determinant
//! - `inv()` - Matrix inverse  
//! - `pinv()` - Moore-Penrose pseudoinverse
//! - `norm()` - Matrix and vector norms
//! - `cond()` - Condition number
//! - `matrix_rank()` - Matrix rank
//!
//! ### Matrix Decompositions  
//! - `lu()` - LU decomposition
//! - `qr()` - QR decomposition
//! - `svd()` - Singular value decomposition
//! - `cholesky()` - Cholesky decomposition
//! - `schur()` - Schur decomposition
//! - `polar()` - Polar decomposition
//! - `rq()` - RQ decomposition
//!
//! ### Eigenvalue Problems
//! - `eig()` - General eigenvalue problem
//! - `eigh()` - Symmetric/Hermitian eigenvalue problem
//! - `eigvals()` - Eigenvalues only
//! - `eigvalsh()` - Eigenvalues of symmetric/Hermitian matrix
//!
//! ### Linear System Solvers
//! - `solve()` - General linear system solver
//! - `lstsq()` - Least squares solver
//! - `solve_triangular()` - Triangular system solver
//! - `solve_banded()` - Banded matrix solver
//!
//! ### Matrix Functions
//! - `expm()` - Matrix exponential
//! - `logm()` - Matrix logarithm
//! - `sqrtm()` - Matrix square root
//! - `funm()` - General matrix function
//! - `cosm()`, `sinm()`, `tanm()` - Trigonometric matrix functions
//!
//! ### Utilities
//! - `block_diag()` - Block diagonal matrix construction

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, NumAssign, Zero};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::{basic, decomposition, eigen, matrix_functions, norm as norm_mod, solve};

/// Type alias for SVD decomposition result
pub type SvdResult<F> = (Option<Array2<F>>, Array1<F>, Option<Array2<F>>);

/// Type alias for least squares solver result
pub type LstsqResult<F> = (Array2<F>, Option<Array1<F>>, usize, Array1<F>);

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
    basic::det(a, None)
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
    basic::inv(a, None)
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
        let eigenvalues = eigen::eigvalsh(a, None)?;
        Ok((eigenvalues, None))
    } else {
        let (eigenvalues, eigenvectors) = eigen::eigh(a, None)?;
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
    decomposition::lu(a, None)
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
            let (q, r) = decomposition::qr(a, None)?;
            Ok((Some(q), r))
        }
        "r" => {
            let (_, r) = decomposition::qr(a, None)?;
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
        let (u, s, vt) = decomposition::svd(a, full_matrices, None)?;
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
    let l = decomposition::cholesky(a, None)?;

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
    solve::solve_multiple(a, b, None)
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
// Re-export with SciPy-compatible wrapper
// pub use crate::matrix_functions::sqrtm;

/// Matrix square root (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `check_finite` - Whether to check that input contains only finite numbers (optional, defaults to true)
///
/// # Returns
/// * Matrix square root of a
pub fn sqrtm<F>(a: &ArrayView2<F>, check_finite: Option<bool>) -> LinalgResult<Array2<F>>
where
    F: Float + Sum + NumAssign + ndarray::ScalarOperand,
{
    let check = check_finite.unwrap_or(true);
    if check {
        for &elem in a.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Matrix contains non-finite values".to_string(),
                ));
            }
        }
    }

    // Use default parameters: max_iter=100, tol=1e-12
    matrix_functions::sqrtm(a, 100, F::from(1e-12).unwrap())
}

/// Compute matrix or vector norm (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input array (1-D or 2-D)
/// * `ord` - Order of the norm (None, 'fro', 'nuc', 1, -1, 2, -2, inf, -inf)
/// * `axis` - Axis along which to compute norm for 2-D arrays
/// * `keepdims` - Whether to keep dimensions in output
/// * `check_finite` - Whether to check that input contains only finite numbers
///
/// # Returns
/// * Norm of the array
pub fn norm<F>(
    a: &ArrayView2<F>,
    ord: Option<&str>,
    axis: Option<usize>,
    _keepdims: bool,
    check_finite: bool,
) -> LinalgResult<F>
where
    F: Float + Sum + NumAssign + ndarray::ScalarOperand,
{
    if check_finite {
        for &elem in a.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Array contains non-finite values".to_string(),
                ));
            }
        }
    }

    match (ord, axis) {
        (None, None) | (Some("fro"), None) => norm_mod::matrix_norm(a, "frobenius", None),
        (Some("nuc"), None) => Err(LinalgError::NotImplementedError(
            "Nuclear norm not yet implemented".to_string(),
        )),
        (Some("1"), None) => norm_mod::matrix_norm(a, "1", None),
        (Some("-1"), None) => norm_mod::matrix_norm(a, "-1", None),
        (Some("2"), None) => norm_mod::matrix_norm(a, "2", None),
        (Some("-2"), None) => norm_mod::matrix_norm(a, "-2", None),
        (Some("inf"), None) => norm_mod::matrix_norm(a, "inf", None),
        (Some("-inf"), None) => norm_mod::matrix_norm(a, "-inf", None),
        _ => Err(LinalgError::NotImplementedError(
            "Specified norm parameters not yet implemented".to_string(),
        )),
    }
}

/// Compute vector norm (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input 1-D array
/// * `ord` - Order of the norm
/// * `check_finite` - Whether to check that input contains only finite numbers
///
/// # Returns
/// * Norm of the vector
pub fn vector_norm<F>(a: &ArrayView1<F>, ord: Option<f64>, check_finite: bool) -> LinalgResult<F>
where
    F: Float + Sum + NumAssign,
{
    if check_finite {
        for &elem in a.iter() {
            if !elem.is_finite() {
                return Err(LinalgError::ValueError(
                    "Array contains non-finite values".to_string(),
                ));
            }
        }
    }

    match ord {
        None | Some(2.0) => norm_mod::vector_norm(a, 2),
        Some(1.0) => norm_mod::vector_norm(a, 1),
        Some(p) if p.is_infinite() && p > 0.0 => {
            // Infinity norm (max absolute value)
            let max_abs = a.iter().map(|&x| x.abs()).fold(F::zero(), F::max);
            Ok(max_abs)
        }
        Some(p) if p.is_infinite() && p < 0.0 => {
            // Negative infinity norm (min absolute value)
            let min_abs = a.iter().map(|&x| x.abs()).fold(F::infinity(), F::min);
            Ok(min_abs)
        }
        Some(0.0) => {
            // 0-norm (count of non-zero elements)
            let count = a.iter().filter(|&&x| x != F::zero()).count();
            Ok(F::from(count).unwrap())
        }
        Some(p) => {
            // General p-norm
            let sum: F = a
                .iter()
                .map(|&x| x.abs().powf(F::from(p).unwrap()))
                .fold(F::zero(), |acc, x| acc + x);
            Ok(sum.powf(F::one() / F::from(p).unwrap()))
        }
    }
}

/// Compute Moore-Penrose pseudoinverse (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `rcond` - Cutoff for small singular values
/// * `hermitian` - Whether to treat as Hermitian matrix
/// * `check_finite` - Whether to check that input contains only finite numbers
///
/// # Returns
/// * Pseudoinverse of the matrix
pub fn pinv<F>(
    a: &ArrayView2<F>,
    rcond: Option<F>,
    _hermitian: bool,
    check_finite: bool,
) -> LinalgResult<Array2<F>>
where
    F: Float + Sum + NumAssign + ndarray::ScalarOperand,
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

    // Use SVD to compute pseudoinverse
    let (u, s, vt) = decomposition::svd(a, false, None)?;

    // Determine cutoff threshold
    let threshold = rcond.unwrap_or_else(|| {
        let max_singular_value = s.iter().cloned().fold(F::zero(), F::max);
        max_singular_value * F::from(1e-15).unwrap() * F::from(a.dim().0.max(a.dim().1)).unwrap()
    });

    // Compute reciprocal of singular values above threshold
    let s_inv: Array1<F> = s.mapv(|val| {
        if val > threshold {
            F::one() / val
        } else {
            F::zero()
        }
    });

    // Reconstruct pseudoinverse: A+ = V * S+ * U^T
    let vs_inv = vt.t().dot(&Array2::from_diag(&s_inv));
    Ok(vs_inv.dot(&u.t()))
}

/// Compute condition number (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `p` - Order of the norm (None, 1, -1, 2, -2, 'fro', 'inf', '-inf')
///
/// # Returns
/// * Condition number of the matrix
pub fn cond<F>(a: &ArrayView2<F>, p: Option<&str>) -> LinalgResult<F>
where
    F: Float + Sum + NumAssign + ndarray::ScalarOperand,
{
    match p {
        None | Some("2") => norm_mod::cond(a, p, None),
        Some("1") => {
            let norm_a = norm_mod::matrix_norm(a, "1", None)?;
            let inv_a = basic::inv(a, None)?;
            let norm_inv_a = norm_mod::matrix_norm(&inv_a.view(), "1", None)?;
            Ok(norm_a * norm_inv_a)
        }
        Some("inf") => {
            let norm_a = norm_mod::matrix_norm(a, "inf", None)?;
            let inv_a = basic::inv(a, None)?;
            let norm_inv_a = norm_mod::matrix_norm(&inv_a.view(), "inf", None)?;
            Ok(norm_a * norm_inv_a)
        }
        Some("fro") => {
            let norm_a = norm_mod::matrix_norm(a, "frobenius", None)?;
            let inv_a = basic::inv(a, None)?;
            let norm_inv_a = norm_mod::matrix_norm(&inv_a.view(), "frobenius", None)?;
            Ok(norm_a * norm_inv_a)
        }
        _ => Err(LinalgError::InvalidInput(format!(
            "Unsupported norm type for condition number: {:?}",
            p
        ))),
    }
}

/// Compute matrix rank (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `tol` - Threshold below which singular values are considered zero
/// * `hermitian` - Whether to treat as Hermitian matrix
/// * `check_finite` - Whether to check that input contains only finite numbers
///
/// # Returns
/// * Rank of the matrix
pub fn matrix_rank<F>(
    a: &ArrayView2<F>,
    tol: Option<F>,
    _hermitian: bool,
    check_finite: bool,
) -> LinalgResult<usize>
where
    F: Float + Sum + NumAssign + ndarray::ScalarOperand,
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

    norm_mod::matrix_rank(a, tol, None)
}

/// Solve linear least squares problem (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Coefficient matrix
/// * `b` - Right-hand side
/// * `cond` - Cutoff for rank determination
/// * `overwrite_a` - Allow overwriting data in `a` (currently ignored)
/// * `overwrite_b` - Allow overwriting data in `b` (currently ignored)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
/// * `lapack_driver` - LAPACK routine to use (currently ignored)
///
/// # Returns
/// * Tuple of (solution, residuals, rank, singular_values)
#[allow(clippy::too_many_arguments)]
pub fn lstsq<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    cond: Option<F>,
    _overwrite_a: bool,
    _overwrite_b: bool,
    check_finite: bool,
    _lapack_driver: Option<&str>,
) -> LinalgResult<LstsqResult<F>>
where
    F: Float + Sum + NumAssign + ndarray::ScalarOperand,
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

    // Convert 2D b to 1D for our lstsq function
    let b_1d = if b.ncols() == 1 {
        b.column(0).to_owned()
    } else {
        return Err(LinalgError::InvalidInput(
            "Multiple right-hand sides not yet supported".to_string(),
        ));
    };
    let lstsq_result = solve::lstsq(a, &b_1d.view(), None)?;

    // Compute SVD to get rank and singular values
    let (_, s, _) = decomposition::svd(a, false, None)?;

    // Determine rank based on condition number
    let threshold = cond.unwrap_or_else(|| {
        let max_sv = s.iter().cloned().fold(F::zero(), F::max);
        max_sv * F::from(1e-15).unwrap() * F::from(a.dim().0.max(a.dim().1)).unwrap()
    });

    let rank = s.iter().filter(|&&val| val > threshold).count();

    // Clone the solution before moving it
    let solution_1d = lstsq_result.x.clone();
    let solution_2d = lstsq_result.x.insert_axis(Axis(1));

    // Compute residuals if overdetermined system
    let residuals = if a.nrows() > a.ncols() {
        let ax = a.dot(&solution_1d);
        let residual_vec = &b_1d - &ax;
        let residual_sum = residual_vec
            .iter()
            .map(|&x| x * x)
            .fold(F::zero(), |acc, x| acc + x);
        Some(Array1::from_elem(1, residual_sum))
    } else {
        None
    };

    Ok((solution_2d, residuals, rank, s))
}

/// Solve triangular system (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Triangular matrix
/// * `b` - Right-hand side
/// * `lower` - Whether `a` is lower triangular
/// * `trans` - Type of system to solve (0: Ax=b, 1: A^Tx=b, 2: A^Hx=b)
/// * `unit_diagonal` - Whether to assume unit diagonal
/// * `overwrite_b` - Allow overwriting data in `b` (currently ignored)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
///
/// # Returns
/// * Solution to the triangular system
#[allow(clippy::too_many_arguments)]
pub fn solve_triangular<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    lower: bool,
    trans: u8,
    unit_diagonal: bool,
    _overwrite_b: bool,
    check_finite: bool,
) -> LinalgResult<Array2<F>>
where
    F: Float + Sum + NumAssign,
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

    if trans != 0 || unit_diagonal {
        return Err(LinalgError::NotImplementedError(
            "Transposed and unit diagonal triangular solve not yet implemented".to_string(),
        ));
    }

    // Convert 2D b to 1D for our solve_triangular function
    let b_1d = if b.ncols() == 1 {
        b.column(0).to_owned()
    } else {
        return Err(LinalgError::InvalidInput(
            "Multiple right-hand sides not yet supported for triangular solve".to_string(),
        ));
    };

    let result_1d = solve::solve_triangular(a, &b_1d.view(), lower, false)?;
    Ok(result_1d.insert_axis(Axis(1)))
}

/// RQ decomposition (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `overwrite_a` - Allow overwriting data in `a` (currently ignored)
/// * `lwork` - Work array size (currently ignored)
/// * `mode` - Decomposition mode ('full' or 'economic')
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
///
/// # Returns
/// * Tuple of (R, Q)
pub fn rq<F>(
    a: &ArrayView2<F>,
    _overwrite_a: bool,
    _lwork: Option<usize>,
    mode: &str,
    check_finite: bool,
) -> LinalgResult<(Array2<F>, Array2<F>)>
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

    if mode != "full" && mode != "economic" {
        return Err(LinalgError::InvalidInput(format!(
            "Invalid RQ mode: {}",
            mode
        )));
    }

    // RQ decomposition can be computed using QR decomposition of the transpose
    // If A^T = Q₁R₁ (QR decomposition), then A = R₁^T Q₁^T = RQ
    // where R = R₁^T (upper triangular) and Q = Q₁^T (orthogonal)

    // Step 1: Transpose the input matrix
    let a_transpose = a.t().to_owned();

    // Step 2: Compute QR decomposition of A^T
    let (q_temp, r_temp) = decomposition::qr(&a_transpose.view(), None)?;

    // Step 3: Get RQ decomposition: R = R₁^T, Q = Q₁^T
    let r = r_temp.t().to_owned(); // R₁^T is upper triangular
    let q = q_temp.t().to_owned(); // Q₁^T is orthogonal

    Ok((r, q))
}

/// Polar decomposition (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `side` - Whether to compute left ('left') or right ('right') polar decomposition
///
/// # Returns
/// * Tuple of (U, P) for right polar or (P, U) for left polar
pub fn polar<F>(a: &ArrayView2<F>, side: &str) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + Sum + NumAssign + ndarray::ScalarOperand,
{
    match side {
        "right" => {
            // A = UP where U is unitary and P is positive semidefinite
            let (u, s, vt) = decomposition::svd(a, false, None)?;
            let unitary = u.dot(&vt);
            let positive = vt.t().dot(&Array2::from_diag(&s)).dot(&vt);
            Ok((unitary, positive))
        }
        "left" => {
            // A = PU where P is positive semidefinite and U is unitary
            let (u, s, vt) = decomposition::svd(a, false, None)?;
            let unitary = u.dot(&vt);
            let positive = u.dot(&Array2::from_diag(&s)).dot(&u.t());
            Ok((positive, unitary))
        }
        _ => Err(LinalgError::InvalidInput(format!(
            "Invalid polar decomposition side: {}",
            side
        ))),
    }
}

/// General matrix function (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `func` - Function to apply (currently limited to common functions)
/// * `disp` - Whether to display warnings (currently ignored)
///
/// # Returns
/// * Matrix function result
pub fn funm<F>(a: &ArrayView2<F>, func: &str, _disp: bool) -> LinalgResult<Array2<F>>
where
    F: Float + Sum + NumAssign + ndarray::ScalarOperand,
{
    match func {
        "exp" => matrix_functions::expm(a, None),
        "log" => matrix_functions::logm(a),
        "sqrt" => matrix_functions::sqrtm(a, 100, F::from(1e-12).unwrap()),
        "cos" => cosm(a),
        "sin" => sinm(a),
        "tan" => tanm(a),
        _ => Err(LinalgError::NotImplementedError(format!(
            "Matrix function '{}' not yet implemented",
            func
        ))),
    }
}

/// Matrix cosine (SciPy-compatible interface)
pub fn cosm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + Sum + NumAssign + ndarray::ScalarOperand,
{
    matrix_functions::cosm(a)
}

/// Matrix sine (SciPy-compatible interface)
pub fn sinm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + Sum + NumAssign + ndarray::ScalarOperand,
{
    matrix_functions::sinm(a)
}

/// Matrix tangent (SciPy-compatible interface)
pub fn tanm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + Sum + NumAssign + ndarray::ScalarOperand,
{
    matrix_functions::tanm(a)
}

/// Schur decomposition (SciPy-compatible interface)
///
/// # Arguments
/// * `a` - Input matrix
/// * `output` - Type of output ('real' or 'complex')
/// * `lwork` - Work array size (currently ignored)
/// * `overwrite_a` - Allow overwriting data in `a` (currently ignored)
/// * `sort` - Sorting function for eigenvalues (currently ignored)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
///
/// # Returns
/// * Tuple of (T, Z) where A = Z T Z^H
#[allow(clippy::too_many_arguments)]
pub fn schur<F>(
    a: &ArrayView2<F>,
    output: &str,
    _lwork: Option<usize>,
    _overwrite_a: bool,
    _sort: Option<fn(F) -> bool>,
    check_finite: bool,
) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + Sum + NumAssign + 'static,
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

    match output {
        "real" | "complex" => {
            // Use the actual schur decomposition implementation
            decomposition::schur(a)
        }
        _ => Err(LinalgError::InvalidInput(format!(
            "Invalid Schur output type: {}",
            output
        ))),
    }
}

/// Create block diagonal matrix (SciPy-compatible interface)
///
/// # Arguments
/// * `arrays` - List of 2-D arrays to place on diagonal
///
/// # Returns
/// * Block diagonal matrix
pub fn block_diag<F>(arrays: &[ArrayView2<F>]) -> LinalgResult<Array2<F>>
where
    F: Float + Zero,
{
    if arrays.is_empty() {
        return Err(LinalgError::InvalidInput(
            "Cannot create block diagonal matrix from empty array list".to_string(),
        ));
    }

    // Calculate total dimensions
    let total_rows: usize = arrays.iter().map(|a| a.nrows()).sum();
    let total_cols: usize = arrays.iter().map(|a| a.ncols()).sum();

    let mut result = Array2::zeros((total_rows, total_cols));

    let mut row_offset = 0;
    let mut col_offset = 0;

    for array in arrays {
        let (rows, cols) = array.dim();
        result
            .slice_mut(ndarray::s![
                row_offset..row_offset + rows,
                col_offset..col_offset + cols
            ])
            .assign(array);
        row_offset += rows;
        col_offset += cols;
    }

    Ok(result)
}

/// Solve banded linear system (SciPy-compatible interface)
///
/// # Arguments
/// * `ab` - Banded matrix in LAPACK format
/// * `b` - Right-hand side
/// * `overwrite_ab` - Allow overwriting data in `ab` (currently ignored)
/// * `overwrite_b` - Allow overwriting data in `b` (currently ignored)
/// * `check_finite` - Whether to check that input matrices contain only finite numbers
///
/// # Returns
/// * Solution to the banded system
pub fn solve_banded<F>(
    _ab: &ArrayView2<F>,
    _b: &ArrayView2<F>,
    _overwrite_ab: bool,
    _overwrite_b: bool,
    check_finite: bool,
) -> LinalgResult<Array2<F>>
where
    F: Float + Sum + NumAssign,
{
    if check_finite {
        // Check would go here
    }

    Err(LinalgError::NotImplementedError(
        "Banded matrix solver not yet implemented".to_string(),
    ))
}
