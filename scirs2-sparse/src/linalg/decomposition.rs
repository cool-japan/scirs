//! Matrix decomposition algorithms for sparse matrices
//!
//! This module provides various matrix decomposition algorithms optimized
//! for sparse matrices, including LU, QR, Cholesky, and incomplete variants.

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

/// LU decomposition result
#[derive(Debug, Clone)]
pub struct LUResult<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Lower triangular factor
    pub l: CsrArray<T>,
    /// Upper triangular factor
    pub u: CsrArray<T>,
    /// Permutation matrix (as permutation vector)
    pub p: Array1<usize>,
    /// Whether decomposition was successful
    pub success: bool,
}

/// QR decomposition result
#[derive(Debug, Clone)]
pub struct QRResult<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Orthogonal factor Q
    pub q: CsrArray<T>,
    /// Upper triangular factor R
    pub r: CsrArray<T>,
    /// Whether decomposition was successful
    pub success: bool,
}

/// Cholesky decomposition result
#[derive(Debug, Clone)]
pub struct CholeskyResult<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Lower triangular Cholesky factor
    pub l: CsrArray<T>,
    /// Whether decomposition was successful
    pub success: bool,
}

/// Pivoted Cholesky decomposition result
#[derive(Debug, Clone)]
pub struct PivotedCholeskyResult<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Lower triangular Cholesky factor
    pub l: CsrArray<T>,
    /// Permutation matrix (as permutation vector)
    pub p: Array1<usize>,
    /// Rank of the decomposition (number of positive eigenvalues)
    pub rank: usize,
    /// Whether decomposition was successful
    pub success: bool,
}

/// Pivoting strategy for LU decomposition
#[derive(Debug, Clone, Default)]
pub enum PivotingStrategy {
    /// No pivoting (fastest but potentially unstable)
    None,
    /// Partial pivoting - choose largest element in column (default)
    #[default]
    Partial,
    /// Threshold pivoting - partial pivoting with threshold
    Threshold(f64),
    /// Scaled partial pivoting - account for row scaling
    ScaledPartial,
    /// Complete pivoting - choose largest element in submatrix (most stable but expensive)
    Complete,
    /// Rook pivoting - hybrid approach balancing stability and cost
    Rook,
}

/// Options for LU decomposition
#[derive(Debug, Clone)]
pub struct LUOptions {
    /// Pivoting strategy to use
    pub pivoting: PivotingStrategy,
    /// Threshold for numerical zero (default: 1e-14)
    pub zero_threshold: f64,
    /// Whether to check for singularity (default: true)
    pub check_singular: bool,
}

impl Default for LUOptions {
    fn default() -> Self {
        Self {
            pivoting: PivotingStrategy::default(),
            zero_threshold: 1e-14,
            check_singular: true,
        }
    }
}

/// Options for incomplete LU decomposition
#[derive(Debug, Clone)]
pub struct ILUOptions {
    /// Drop tolerance for numerical stability
    pub drop_tol: f64,
    /// Fill factor (maximum fill-in ratio)
    pub fill_factor: f64,
    /// Maximum number of fill-in entries per row
    pub max_fill_per_row: usize,
    /// Pivoting strategy to use
    pub pivoting: PivotingStrategy,
}

impl Default for ILUOptions {
    fn default() -> Self {
        Self {
            drop_tol: 1e-4,
            fill_factor: 2.0,
            max_fill_per_row: 20,
            pivoting: PivotingStrategy::default(),
        }
    }
}

/// Options for incomplete Cholesky decomposition
#[derive(Debug, Clone)]
pub struct ICOptions {
    /// Drop tolerance for numerical stability
    pub drop_tol: f64,
    /// Fill factor (maximum fill-in ratio)
    pub fill_factor: f64,
    /// Maximum number of fill-in entries per row
    pub max_fill_per_row: usize,
}

impl Default for ICOptions {
    fn default() -> Self {
        Self {
            drop_tol: 1e-4,
            fill_factor: 2.0,
            max_fill_per_row: 20,
        }
    }
}

/// Compute sparse LU decomposition with partial pivoting (backward compatibility)
///
/// Computes the LU decomposition of a sparse matrix A such that P*A = L*U,
/// where P is a permutation matrix, L is lower triangular, and U is upper triangular.
///
/// # Arguments
///
/// * `matrix` - The sparse matrix to decompose
/// * `pivot_threshold` - Pivoting threshold for numerical stability (0.0 to 1.0)
///
/// # Returns
///
/// LU decomposition result
///
/// # Examples
///
/// ```
/// use scirs2_sparse::linalg::lu_decomposition;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a sparse matrix
/// let rows = vec![0, 0, 1, 2];
/// let cols = vec![0, 1, 1, 2];
/// let data = vec![2.0, 1.0, 3.0, 4.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// let lu_result = lu_decomposition(&matrix, 0.1).unwrap();
/// ```
#[allow(dead_code)]
pub fn lu_decomposition<T, S>(_matrix: &S, pivotthreshold: f64) -> SparseResult<LUResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    // Use _threshold pivoting for backward compatibility
    let options = LUOptions {
        pivoting: PivotingStrategy::Threshold(pivotthreshold),
        zero_threshold: 1e-14,
        check_singular: true,
    };

    lu_decomposition_with_options(_matrix, Some(options))
}

/// Compute sparse LU decomposition with enhanced pivoting strategies
///
/// Computes the LU decomposition of a sparse matrix A such that P*A = L*U,
/// where P is a permutation matrix, L is lower triangular, and U is upper triangular.
/// This version supports multiple pivoting strategies for enhanced numerical stability.
///
/// # Arguments
///
/// * `matrix` - The sparse matrix to decompose
/// * `options` - LU decomposition options (pivoting strategy, thresholds, etc.)
///
/// # Returns
///
/// LU decomposition result
///
/// # Examples
///
/// ```
/// use scirs2_sparse::linalg::{lu_decomposition_with_options, LUOptions, PivotingStrategy};
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a sparse matrix
/// let rows = vec![0, 0, 1, 2];
/// let cols = vec![0, 1, 1, 2];
/// let data = vec![2.0, 1.0, 3.0, 4.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// let options = LUOptions {
///     pivoting: PivotingStrategy::ScaledPartial,
///     zero_threshold: 1e-12,
///     check_singular: true,
/// };
/// let lu_result = lu_decomposition_with_options(&matrix, Some(options)).unwrap();
/// ```
#[allow(dead_code)]
pub fn lu_decomposition_with_options<T, S>(
    matrix: &S,
    options: Option<LUOptions>,
) -> SparseResult<LUResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    let opts = options.unwrap_or_default();
    let (n, m) = matrix.shape();
    if n != m {
        return Err(SparseError::ValueError(
            "Matrix must be square for LU decomposition".to_string(),
        ));
    }

    // Convert to working format
    let (row_indices, col_indices, values) = matrix.find();
    let mut working_matrix = SparseWorkingMatrix::from_triplets(
        row_indices.as_slice().unwrap(),
        col_indices.as_slice().unwrap(),
        values.as_slice().unwrap(),
        n,
    );

    // Initialize permutations
    let mut row_perm: Vec<usize> = (0..n).collect();
    let mut col_perm: Vec<usize> = (0..n).collect();

    // Compute row scaling factors for scaled partial pivoting
    let mut row_scales = vec![T::one(); n];
    if matches!(opts.pivoting, PivotingStrategy::ScaledPartial) {
        for (i, scale) in row_scales.iter_mut().enumerate().take(n) {
            let row_data = working_matrix.get_row(i);
            let max_val =
                row_data
                    .values()
                    .map(|&v| v.abs())
                    .fold(T::zero(), |a, b| if a > b { a } else { b });
            if max_val > T::zero() {
                *scale = max_val;
            }
        }
    }

    // Gaussian elimination with enhanced pivoting
    for k in 0..n - 1 {
        // Find pivot using selected strategy
        let (pivot_row, pivot_col) =
            find_enhanced_pivot(&working_matrix, k, &row_perm, &col_perm, &row_scales, &opts)?;

        // Apply row and column permutations
        if pivot_row != k {
            row_perm.swap(k, pivot_row);
        }
        if pivot_col != k
            && matches!(
                opts.pivoting,
                PivotingStrategy::Complete | PivotingStrategy::Rook
            )
        {
            col_perm.swap(k, pivot_col);
            // When columns are swapped, we need to update all matrix elements
            for &row_idx in row_perm.iter().take(n) {
                let temp = working_matrix.get(row_idx, k);
                working_matrix.set(row_idx, k, working_matrix.get(row_idx, pivot_col));
                working_matrix.set(row_idx, pivot_col, temp);
            }
        }

        let actual_pivot_row = row_perm[k];
        let actual_pivot_col = col_perm[k];
        let pivot_value = working_matrix.get(actual_pivot_row, actual_pivot_col);

        // Check for numerical singularity
        if opts.check_singular && pivot_value.abs() < T::from(opts.zero_threshold).unwrap() {
            return Ok(LUResult {
                l: CsrArray::from_triplets(&[], &[], &[], (n, n), false)?,
                u: CsrArray::from_triplets(&[], &[], &[], (n, n), false)?,
                p: Array1::from_vec(row_perm),
                success: false,
            });
        }

        // Eliminate below pivot
        for &actual_row_i in row_perm.iter().take(n).skip(k + 1) {
            let factor = working_matrix.get(actual_row_i, actual_pivot_col) / pivot_value;

            if !factor.is_zero() {
                // Store multiplier in L
                working_matrix.set(actual_row_i, actual_pivot_col, factor);

                // Update row i
                let pivot_row_data = working_matrix.get_row(actual_pivot_row);
                for (col, &value) in &pivot_row_data {
                    if *col > k {
                        let old_val = working_matrix.get(actual_row_i, *col);
                        working_matrix.set(actual_row_i, *col, old_val - factor * value);
                    }
                }
            }
        }
    }

    // Extract L and U matrices with proper permutation
    let (l_rows, l_cols, l_vals, u_rows, u_cols, u_vals) =
        extract_lu_factors(&working_matrix, &row_perm, n);

    let l = CsrArray::from_triplets(&l_rows, &l_cols, &l_vals, (n, n), false)?;
    let u = CsrArray::from_triplets(&u_rows, &u_cols, &u_vals, (n, n), false)?;

    Ok(LUResult {
        l,
        u,
        p: Array1::from_vec(row_perm),
        success: true,
    })
}

/// Compute sparse QR decomposition using Givens rotations
///
/// Computes the QR decomposition of a sparse matrix A = Q*R,
/// where Q is orthogonal and R is upper triangular.
///
/// # Arguments
///
/// * `matrix` - The sparse matrix to decompose
///
/// # Returns
///
/// QR decomposition result
///
/// # Examples
///
/// ```
/// use scirs2_sparse::linalg::qr_decomposition;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// let rows = vec![0, 1, 2];
/// let cols = vec![0, 0, 1];
/// let data = vec![1.0, 2.0, 3.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 2), false).unwrap();
///
/// let qr_result = qr_decomposition(&matrix).unwrap();
/// ```
#[allow(dead_code)]
pub fn qr_decomposition<T, S>(matrix: &S) -> SparseResult<QRResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    let (m, n) = matrix.shape();

    // Convert to dense for QR (sparse QR is complex)
    let dense_matrix = matrix.to_array();

    // Simple Gram-Schmidt QR decomposition
    let mut q = Array2::zeros((m, n));
    let mut r = Array2::zeros((n, n));

    for j in 0..n {
        // Copy column j
        for i in 0..m {
            q[[i, j]] = dense_matrix[[i, j]];
        }

        // Orthogonalize against previous columns
        for k in 0..j {
            let mut dot = T::zero();
            for i in 0..m {
                dot = dot + q[[i, k]] * dense_matrix[[i, j]];
            }
            r[[k, j]] = dot;

            for i in 0..m {
                q[[i, j]] = q[[i, j]] - dot * q[[i, k]];
            }
        }

        // Normalize
        let mut norm = T::zero();
        for i in 0..m {
            norm = norm + q[[i, j]] * q[[i, j]];
        }
        norm = norm.sqrt();
        r[[j, j]] = norm;

        if !norm.is_zero() {
            for i in 0..m {
                q[[i, j]] = q[[i, j]] / norm;
            }
        }
    }

    // Convert back to sparse
    let q_sparse = dense_to_sparse(&q)?;
    let r_sparse = dense_to_sparse(&r)?;

    Ok(QRResult {
        q: q_sparse,
        r: r_sparse,
        success: true,
    })
}

/// Compute sparse Cholesky decomposition
///
/// Computes the Cholesky decomposition of a symmetric positive definite matrix A = L*L^T,
/// where L is lower triangular.
///
/// # Arguments
///
/// * `matrix` - The symmetric positive definite sparse matrix
///
/// # Returns
///
/// Cholesky decomposition result
///
/// # Examples
///
/// ```
/// use scirs2_sparse::linalg::cholesky_decomposition;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a simple SPD matrix
/// let rows = vec![0, 1, 1, 2, 2, 2];
/// let cols = vec![0, 0, 1, 0, 1, 2];
/// let data = vec![4.0, 2.0, 5.0, 1.0, 3.0, 6.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// let chol_result = cholesky_decomposition(&matrix).unwrap();
/// ```
#[allow(dead_code)]
pub fn cholesky_decomposition<T, S>(matrix: &S) -> SparseResult<CholeskyResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    let (n, m) = matrix.shape();
    if n != m {
        return Err(SparseError::ValueError(
            "Matrix must be square for Cholesky decomposition".to_string(),
        ));
    }

    // Convert to working format
    let (row_indices, col_indices, values) = matrix.find();
    let mut working_matrix = SparseWorkingMatrix::from_triplets(
        row_indices.as_slice().unwrap(),
        col_indices.as_slice().unwrap(),
        values.as_slice().unwrap(),
        n,
    );

    // Cholesky decomposition algorithm
    for k in 0..n {
        // Compute diagonal element
        let mut sum = T::zero();
        for j in 0..k {
            let l_kj = working_matrix.get(k, j);
            sum = sum + l_kj * l_kj;
        }

        let a_kk = working_matrix.get(k, k);
        let diag_val = a_kk - sum;

        if diag_val <= T::zero() {
            return Ok(CholeskyResult {
                l: CsrArray::from_triplets(&[], &[], &[], (n, n), false)?,
                success: false,
            });
        }

        let l_kk = diag_val.sqrt();
        working_matrix.set(k, k, l_kk);

        // Compute below-diagonal elements
        for i in (k + 1)..n {
            let mut sum = T::zero();
            for j in 0..k {
                sum = sum + working_matrix.get(i, j) * working_matrix.get(k, j);
            }

            let a_ik = working_matrix.get(i, k);
            let l_ik = (a_ik - sum) / l_kk;
            working_matrix.set(i, k, l_ik);
        }
    }

    // Extract lower triangular _matrix
    let (l_rows, l_cols, l_vals) = extract_lower_triangular(&working_matrix, n);
    let l = CsrArray::from_triplets(&l_rows, &l_cols, &l_vals, (n, n), false)?;

    Ok(CholeskyResult { l, success: true })
}

/// Compute pivoted Cholesky decomposition
///
/// Computes the pivoted Cholesky decomposition of a symmetric matrix A = P^T * L * L^T * P,
/// where P is a permutation matrix and L is lower triangular. This version can handle
/// indefinite matrices by determining the rank and producing a partial decomposition.
///
/// # Arguments
///
/// * `matrix` - The symmetric sparse matrix
/// * `threshold` - Pivoting threshold for numerical stability (default: 1e-12)
///
/// # Returns
///
/// Pivoted Cholesky decomposition result with rank determination
///
/// # Examples
///
/// ```
/// use scirs2_sparse::linalg::pivoted_cholesky_decomposition;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a symmetric indefinite matrix
/// let rows = vec![0, 1, 1, 2, 2, 2];
/// let cols = vec![0, 0, 1, 0, 1, 2];  
/// let data = vec![1.0, 2.0, -1.0, 3.0, 1.0, 2.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// let chol_result = pivoted_cholesky_decomposition(&matrix, Some(1e-12)).unwrap();
/// ```
#[allow(dead_code)]
pub fn pivoted_cholesky_decomposition<T, S>(
    matrix: &S,
    threshold: Option<T>,
) -> SparseResult<PivotedCholeskyResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    let (n, m) = matrix.shape();
    if n != m {
        return Err(SparseError::ValueError(
            "Matrix must be square for Cholesky decomposition".to_string(),
        ));
    }

    let threshold = threshold.unwrap_or_else(|| T::from(1e-12).unwrap());

    // Convert to working format
    let (row_indices, col_indices, values) = matrix.find();
    let mut working_matrix = SparseWorkingMatrix::from_triplets(
        row_indices.as_slice().unwrap(),
        col_indices.as_slice().unwrap(),
        values.as_slice().unwrap(),
        n,
    );

    // Initialize permutation
    let mut perm: Vec<usize> = (0..n).collect();
    let mut rank = 0;

    // Pivoted Cholesky algorithm
    for k in 0..n {
        // Find the pivot: largest diagonal element among remaining
        let mut max_diag = T::zero();
        let mut pivot_idx = k;

        for i in k..n {
            let mut diag_val = working_matrix.get(perm[i], perm[i]);
            for j in 0..k {
                let l_ij = working_matrix.get(perm[i], perm[j]);
                diag_val = diag_val - l_ij * l_ij;
            }
            if diag_val > max_diag {
                max_diag = diag_val;
                pivot_idx = i;
            }
        }

        // Check if we should stop (matrix is not positive definite beyond this point)
        if max_diag <= threshold {
            break;
        }

        // Swap rows/columns in permutation
        if pivot_idx != k {
            perm.swap(k, pivot_idx);
        }

        // Compute L[k,k]
        let l_kk = max_diag.sqrt();
        working_matrix.set(perm[k], perm[k], l_kk);
        rank += 1;

        // Update column k below diagonal
        for i in (k + 1)..n {
            let mut sum = T::zero();
            for j in 0..k {
                sum = sum
                    + working_matrix.get(perm[i], perm[j]) * working_matrix.get(perm[k], perm[j]);
            }

            let a_ik = working_matrix.get(perm[i], perm[k]);
            let l_ik = (a_ik - sum) / l_kk;
            working_matrix.set(perm[i], perm[k], l_ik);
        }
    }

    // Extract lower triangular matrix with proper permutation
    let mut l_rows = Vec::new();
    let mut l_cols = Vec::new();
    let mut l_vals = Vec::new();

    for i in 0..rank {
        for j in 0..=i {
            let val = working_matrix.get(perm[i], perm[j]);
            if val != T::zero() {
                l_rows.push(i);
                l_cols.push(j);
                l_vals.push(val);
            }
        }
    }

    let l = CsrArray::from_triplets(&l_rows, &l_cols, &l_vals, (n, rank), false)?;
    let p = Array1::from_vec(perm);

    Ok(PivotedCholeskyResult {
        l,
        p,
        rank,
        success: true,
    })
}

/// LDLT decomposition result for symmetric indefinite matrices
#[derive(Debug, Clone)]
pub struct LDLTResult<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Lower triangular factor L (unit diagonal)
    pub l: CsrArray<T>,
    /// Diagonal factor D
    pub d: Array1<T>,
    /// Permutation matrix (as permutation vector)
    pub p: Array1<usize>,
    /// Whether decomposition was successful
    pub success: bool,
}

/// Compute LDLT decomposition for symmetric indefinite matrices
///
/// Computes the LDLT decomposition of a symmetric matrix A = P^T * L * D * L^T * P,
/// where P is a permutation matrix, L is unit lower triangular, and D is diagonal.
/// This method can handle indefinite matrices unlike Cholesky decomposition.
///
/// # Arguments
///
/// * `matrix` - The symmetric sparse matrix
/// * `pivoting` - Whether to use pivoting for numerical stability (default: true)
/// * `threshold` - Pivoting threshold for numerical stability (default: 1e-12)
///
/// # Returns
///
/// LDLT decomposition result
///
/// # Examples
///
/// ```
/// use scirs2_sparse::linalg::ldlt_decomposition;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a symmetric indefinite matrix
/// let rows = vec![0, 1, 1, 2, 2, 2];
/// let cols = vec![0, 0, 1, 0, 1, 2];  
/// let data = vec![1.0, 2.0, -1.0, 3.0, 1.0, 2.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// let ldlt_result = ldlt_decomposition(&matrix, Some(true), Some(1e-12)).unwrap();
/// ```
#[allow(dead_code)]
pub fn ldlt_decomposition<T, S>(
    matrix: &S,
    pivoting: Option<bool>,
    threshold: Option<T>,
) -> SparseResult<LDLTResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    let (n, m) = matrix.shape();
    if n != m {
        return Err(SparseError::ValueError(
            "Matrix must be square for LDLT decomposition".to_string(),
        ));
    }

    let use_pivoting = pivoting.unwrap_or(true);
    let threshold = threshold.unwrap_or_else(|| T::from(1e-12).unwrap());

    // Convert to working format
    let (row_indices, col_indices, values) = matrix.find();
    let mut working_matrix = SparseWorkingMatrix::from_triplets(
        row_indices.as_slice().unwrap(),
        col_indices.as_slice().unwrap(),
        values.as_slice().unwrap(),
        n,
    );

    // Initialize permutation
    let mut perm: Vec<usize> = (0..n).collect();
    let mut d_values = vec![T::zero(); n];

    // LDLT decomposition with optional pivoting
    for k in 0..n {
        // Find pivot if pivoting is enabled
        if use_pivoting {
            let pivot_idx = find_ldlt_pivot(&working_matrix, k, &perm, threshold);
            if pivot_idx != k {
                perm.swap(k, pivot_idx);
            }
        }

        let actual_k = perm[k];

        // Compute diagonal element D[k,k]
        let mut diag_val = working_matrix.get(actual_k, actual_k);
        for j in 0..k {
            let l_kj = working_matrix.get(actual_k, perm[j]);
            diag_val = diag_val - l_kj * l_kj * d_values[j];
        }

        d_values[k] = diag_val;

        // Check for numerical issues
        if diag_val.abs() < threshold {
            return Ok(LDLTResult {
                l: CsrArray::from_triplets(&[], &[], &[], (n, n), false)?,
                d: Array1::from_vec(d_values),
                p: Array1::from_vec(perm),
                success: false,
            });
        }

        // Compute column k of L below the diagonal
        for i in (k + 1)..n {
            let actual_i = perm[i];
            let mut l_ik = working_matrix.get(actual_i, actual_k);

            for j in 0..k {
                l_ik = l_ik
                    - working_matrix.get(actual_i, perm[j])
                        * working_matrix.get(actual_k, perm[j])
                        * d_values[j];
            }

            l_ik = l_ik / diag_val;
            working_matrix.set(actual_i, actual_k, l_ik);
        }

        // Set diagonal element of L to 1
        working_matrix.set(actual_k, actual_k, T::one());
    }

    // Extract L matrix (unit lower triangular)
    let (l_rows, l_cols, l_vals) = extract_unit_lower_triangular(&working_matrix, &perm, n);
    let l = CsrArray::from_triplets(&l_rows, &l_cols, &l_vals, (n, n), false)?;

    Ok(LDLTResult {
        l,
        d: Array1::from_vec(d_values),
        p: Array1::from_vec(perm),
        success: true,
    })
}

/// Find pivot for LDLT decomposition using Bunch-Kaufman strategy
#[allow(dead_code)]
fn find_ldlt_pivot<T>(
    matrix: &SparseWorkingMatrix<T>,
    k: usize,
    perm: &[usize],
    threshold: T,
) -> usize
where
    T: Float + Debug + Copy,
{
    let n = matrix.n;
    let mut max_val = T::zero();
    let mut pivot_idx = k;

    // Look for largest diagonal element among remaining rows
    for (i, &actual_i) in perm.iter().enumerate().take(n).skip(k) {
        let diag_val = matrix.get(actual_i, actual_i).abs();

        if diag_val > max_val {
            max_val = diag_val;
            pivot_idx = i;
        }
    }

    // Check if pivot is acceptable
    if max_val >= threshold {
        pivot_idx
    } else {
        k // Use current position if no good pivot found
    }
}

/// Extract unit lower triangular matrix from working matrix
#[allow(dead_code)]
fn extract_unit_lower_triangular<T>(
    matrix: &SparseWorkingMatrix<T>,
    perm: &[usize],
    n: usize,
) -> (Vec<usize>, Vec<usize>, Vec<T>)
where
    T: Float + Debug + Copy,
{
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        let actual_i = perm[i];

        // Add diagonal element (always 1 for unit triangular)
        rows.push(i);
        cols.push(i);
        vals.push(T::one());

        // Add below-diagonal elements
        for (j, &perm_j) in perm.iter().enumerate().take(i) {
            let val = matrix.get(actual_i, perm_j);
            if val != T::zero() {
                rows.push(i);
                cols.push(j);
                vals.push(val);
            }
        }
    }

    (rows, cols, vals)
}

/// Compute incomplete LU decomposition (ILU)
///
/// Computes an approximate LU decomposition with controlled fill-in
/// for use as a preconditioner in iterative methods.
///
/// # Arguments
///
/// * `matrix` - The sparse matrix to decompose
/// * `options` - ILU options controlling fill-in and dropping
///
/// # Returns
///
/// Incomplete LU decomposition result
#[allow(dead_code)]
pub fn incomplete_lu<T, S>(matrix: &S, options: Option<ILUOptions>) -> SparseResult<LUResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    let opts = options.unwrap_or_default();
    let (n, m) = matrix.shape();

    if n != m {
        return Err(SparseError::ValueError(
            "Matrix must be square for ILU decomposition".to_string(),
        ));
    }

    // Convert to working format
    let (row_indices, col_indices, values) = matrix.find();
    let mut working_matrix = SparseWorkingMatrix::from_triplets(
        row_indices.as_slice().unwrap(),
        col_indices.as_slice().unwrap(),
        values.as_slice().unwrap(),
        n,
    );

    // ILU(0) algorithm - no fill-in beyond original sparsity pattern
    for k in 0..n - 1 {
        let pivot_val = working_matrix.get(k, k);

        if pivot_val.abs() < T::from(1e-14).unwrap() {
            continue; // Skip singular pivot
        }

        // Get all non-zero entries in column k below diagonal
        let col_k_entries = working_matrix.get_column_below_diagonal(k);

        for &row_i in &col_k_entries {
            let factor = working_matrix.get(row_i, k) / pivot_val;

            // Drop small factors
            if factor.abs() < T::from(opts.drop_tol).unwrap() {
                working_matrix.set(row_i, k, T::zero());
                continue;
            }

            working_matrix.set(row_i, k, factor);

            // Update row i (only existing non-zeros)
            let row_k_entries = working_matrix.get_row_after_column(k, k);
            for (col_j, &val_kj) in &row_k_entries {
                if working_matrix.has_entry(row_i, *col_j) {
                    let old_val = working_matrix.get(row_i, *col_j);
                    let new_val = old_val - factor * val_kj;

                    // Drop small values
                    if new_val.abs() < T::from(opts.drop_tol).unwrap() {
                        working_matrix.set(row_i, *col_j, T::zero());
                    } else {
                        working_matrix.set(row_i, *col_j, new_val);
                    }
                }
            }
        }
    }

    // Extract L and U factors
    let identity_p: Vec<usize> = (0..n).collect();
    let (l_rows, l_cols, l_vals, u_rows, u_cols, u_vals) =
        extract_lu_factors(&working_matrix, &identity_p, n);

    let l = CsrArray::from_triplets(&l_rows, &l_cols, &l_vals, (n, n), false)?;
    let u = CsrArray::from_triplets(&u_rows, &u_cols, &u_vals, (n, n), false)?;

    Ok(LUResult {
        l,
        u,
        p: Array1::from_vec(identity_p),
        success: true,
    })
}

/// Compute incomplete Cholesky decomposition (IC)
///
/// Computes an approximate Cholesky decomposition with controlled fill-in
/// for use as a preconditioner in iterative methods.
///
/// # Arguments
///
/// * `matrix` - The symmetric positive definite sparse matrix
/// * `options` - IC options controlling fill-in and dropping
///
/// # Returns
///
/// Incomplete Cholesky decomposition result
#[allow(dead_code)]
pub fn incomplete_cholesky<T, S>(
    matrix: &S,
    options: Option<ICOptions>,
) -> SparseResult<CholeskyResult<T>>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    S: SparseArray<T>,
{
    let opts = options.unwrap_or_default();
    let (n, m) = matrix.shape();

    if n != m {
        return Err(SparseError::ValueError(
            "Matrix must be square for IC decomposition".to_string(),
        ));
    }

    // Convert to working format
    let (row_indices, col_indices, values) = matrix.find();
    let mut working_matrix = SparseWorkingMatrix::from_triplets(
        row_indices.as_slice().unwrap(),
        col_indices.as_slice().unwrap(),
        values.as_slice().unwrap(),
        n,
    );

    // IC(0) algorithm - no fill-in beyond original sparsity pattern
    for k in 0..n {
        // Compute diagonal element
        let mut sum = T::zero();
        let row_k_before_k = working_matrix.get_row_before_column(k, k);
        for &val_kj in row_k_before_k.values() {
            sum = sum + val_kj * val_kj;
        }

        let a_kk = working_matrix.get(k, k);
        let diag_val = a_kk - sum;

        if diag_val <= T::zero() {
            return Ok(CholeskyResult {
                l: CsrArray::from_triplets(&[], &[], &[], (n, n), false)?,
                success: false,
            });
        }

        let l_kk = diag_val.sqrt();
        working_matrix.set(k, k, l_kk);

        // Compute below-diagonal elements (only existing entries)
        let col_k_below = working_matrix.get_column_below_diagonal(k);
        for &row_i in &col_k_below {
            let mut sum = T::zero();
            let row_i_before_k = working_matrix.get_row_before_column(row_i, k);
            let row_k_before_k = working_matrix.get_row_before_column(k, k);

            // Compute dot product of L[i, :k] and L[k, :k]
            for (col_j, &val_ij) in &row_i_before_k {
                if let Some(&val_kj) = row_k_before_k.get(col_j) {
                    sum = sum + val_ij * val_kj;
                }
            }

            let a_ik = working_matrix.get(row_i, k);
            let l_ik = (a_ik - sum) / l_kk;

            // Drop small values
            if l_ik.abs() < T::from(opts.drop_tol).unwrap() {
                working_matrix.set(row_i, k, T::zero());
            } else {
                working_matrix.set(row_i, k, l_ik);
            }
        }
    }

    // Extract lower triangular matrix
    let (l_rows, l_cols, l_vals) = extract_lower_triangular(&working_matrix, n);
    let l = CsrArray::from_triplets(&l_rows, &l_cols, &l_vals, (n, n), false)?;

    Ok(CholeskyResult { l, success: true })
}

/// Simple sparse working matrix for decomposition algorithms
struct SparseWorkingMatrix<T>
where
    T: Float + Debug + Copy,
{
    data: HashMap<(usize, usize), T>,
    n: usize,
}

impl<T> SparseWorkingMatrix<T>
where
    T: Float + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    fn from_triplets(rows: &[usize], cols: &[usize], values: &[T], n: usize) -> Self {
        let mut data = HashMap::new();

        for (i, (&row, &col)) in rows.iter().zip(cols.iter()).enumerate() {
            data.insert((row, col), values[i]);
        }

        Self { data, n }
    }

    fn get(&self, row: usize, col: usize) -> T {
        self.data.get(&(row, col)).copied().unwrap_or(T::zero())
    }

    fn set(&mut self, row: usize, col: usize, value: T) {
        if value.is_zero() {
            self.data.remove(&(row, col));
        } else {
            self.data.insert((row, col), value);
        }
    }

    fn has_entry(&self, row: usize, col: usize) -> bool {
        self.data.contains_key(&(row, col))
    }

    fn get_row(&self, row: usize) -> HashMap<usize, T> {
        let mut result = HashMap::new();
        for (&(r, c), &value) in &self.data {
            if r == row {
                result.insert(c, value);
            }
        }
        result
    }

    fn get_row_after_column(&self, row: usize, col: usize) -> HashMap<usize, T> {
        let mut result = HashMap::new();
        for (&(r, c), &value) in &self.data {
            if r == row && c > col {
                result.insert(c, value);
            }
        }
        result
    }

    fn get_row_before_column(&self, row: usize, col: usize) -> HashMap<usize, T> {
        let mut result = HashMap::new();
        for (&(r, c), &value) in &self.data {
            if r == row && c < col {
                result.insert(c, value);
            }
        }
        result
    }

    fn get_column_below_diagonal(&self, col: usize) -> Vec<usize> {
        let mut result = Vec::new();
        for &(r, c) in self.data.keys() {
            if c == col && r > col {
                result.push(r);
            }
        }
        result.sort();
        result
    }
}

/// Find pivot for LU decomposition (backward compatibility)
#[allow(dead_code)]
fn find_pivot<T>(
    matrix: &SparseWorkingMatrix<T>,
    k: usize,
    p: &[usize],
    threshold: f64,
) -> SparseResult<usize>
where
    T: Float + Debug + Copy,
{
    // Use threshold pivoting for backward compatibility
    let opts = LUOptions {
        pivoting: PivotingStrategy::Threshold(threshold),
        zero_threshold: 1e-14,
        check_singular: true,
    };

    let row_scales = vec![T::one(); matrix.n];
    let col_perm: Vec<usize> = (0..matrix.n).collect();

    let (pivot_row, pivot_col) = find_enhanced_pivot(matrix, k, p, &col_perm, &row_scales, &opts)?;
    Ok(pivot_row)
}

/// Enhanced pivoting function supporting multiple strategies
#[allow(dead_code)]
fn find_enhanced_pivot<T>(
    matrix: &SparseWorkingMatrix<T>,
    k: usize,
    row_perm: &[usize],
    col_perm: &[usize],
    row_scales: &[T],
    opts: &LUOptions,
) -> SparseResult<(usize, usize)>
where
    T: Float + Debug + Copy,
{
    let n = matrix.n;

    match &opts.pivoting {
        PivotingStrategy::None => {
            // No pivoting - use diagonal element
            Ok((k, k))
        }

        PivotingStrategy::Partial => {
            // Standard partial pivoting - find largest element in column k
            let mut max_val = T::zero();
            let mut pivot_row = k;

            for (idx, &actual_row) in row_perm.iter().enumerate().skip(k).take(n - k) {
                let i = k + idx;
                let val = matrix.get(actual_row, col_perm[k]).abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = i;
                }
            }

            Ok((pivot_row, k))
        }

        PivotingStrategy::Threshold(threshold) => {
            // Threshold pivoting - use first element above threshold
            let threshold_val = T::from(*threshold).unwrap();
            let mut max_val = T::zero();
            let mut pivot_row = k;

            for (idx, &actual_row) in row_perm.iter().enumerate().skip(k).take(n - k) {
                let i = k + idx;
                let val = matrix.get(actual_row, col_perm[k]).abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = i;
                }
                // Use first element above threshold for efficiency
                if val >= threshold_val {
                    pivot_row = i;
                    break;
                }
            }

            Ok((pivot_row, k))
        }

        PivotingStrategy::ScaledPartial => {
            // Scaled partial pivoting - account for row scaling
            let mut max_ratio = T::zero();
            let mut pivot_row = k;

            for (idx, &actual_row) in row_perm.iter().enumerate().skip(k).take(n - k) {
                let i = k + idx;
                let val = matrix.get(actual_row, col_perm[k]).abs();
                let scale = row_scales[actual_row];

                let ratio = if scale > T::zero() { val / scale } else { val };

                if ratio > max_ratio {
                    max_ratio = ratio;
                    pivot_row = i;
                }
            }

            Ok((pivot_row, k))
        }

        PivotingStrategy::Complete => {
            // Complete pivoting - find largest element in remaining submatrix
            let mut max_val = T::zero();
            let mut pivot_row = k;
            let mut pivot_col = k;

            for (i_idx, &actual_row) in row_perm.iter().enumerate().skip(k).take(n - k) {
                let i = k + i_idx;
                for (j_idx, &actual_col) in col_perm.iter().enumerate().skip(k).take(n - k) {
                    let j = k + j_idx;
                    let val = matrix.get(actual_row, actual_col).abs();
                    if val > max_val {
                        max_val = val;
                        pivot_row = i;
                        pivot_col = j;
                    }
                }
            }

            Ok((pivot_row, pivot_col))
        }

        PivotingStrategy::Rook => {
            // Rook pivoting - alternating row and column searches
            let mut best_row = k;
            let mut best_col = k;
            let mut max_val = T::zero();

            // Start with partial pivoting in column k
            for (idx, &actual_row) in row_perm.iter().enumerate().skip(k).take(n - k) {
                let i = k + idx;
                let val = matrix.get(actual_row, col_perm[k]).abs();
                if val > max_val {
                    max_val = val;
                    best_row = i;
                }
            }

            // If we found a good pivot, check if we can improve by column pivoting
            if max_val > T::from(opts.zero_threshold).unwrap() {
                let actual_best_row = row_perm[best_row];
                let mut col_max = T::zero();

                for (idx, &actual_col) in col_perm.iter().enumerate().skip(k).take(n - k) {
                    let j = k + idx;
                    let val = matrix.get(actual_best_row, actual_col).abs();
                    if val > col_max {
                        col_max = val;
                        best_col = j;
                    }
                }

                // Use column pivot if it's significantly better
                let improvement_threshold = T::from(1.5).unwrap();
                if col_max > max_val * improvement_threshold {
                    // Recompute row pivot for the new column
                    max_val = T::zero();
                    for (idx, &actual_row) in row_perm.iter().enumerate().skip(k).take(n - k) {
                        let i = k + idx;
                        let val = matrix.get(actual_row, col_perm[best_col]).abs();
                        if val > max_val {
                            max_val = val;
                            best_row = i;
                        }
                    }
                }
            }

            Ok((best_row, best_col))
        }
    }
}

/// Extract L and U factors from working matrix
type LuFactors<T> = (
    Vec<usize>, // L row pointers
    Vec<usize>, // L column indices
    Vec<T>,     // L values
    Vec<usize>, // U row pointers
    Vec<usize>, // U column indices
    Vec<T>,     // U values
);

#[allow(dead_code)]
fn extract_lu_factors<T>(matrix: &SparseWorkingMatrix<T>, p: &[usize], n: usize) -> LuFactors<T>
where
    T: Float + Debug + Copy,
{
    let mut l_rows = Vec::new();
    let mut l_cols = Vec::new();
    let mut l_vals = Vec::new();
    let mut u_rows = Vec::new();
    let mut u_cols = Vec::new();
    let mut u_vals = Vec::new();

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let actual_row = p[i];

        // Add diagonal 1 to L
        l_rows.push(i);
        l_cols.push(i);
        l_vals.push(T::one());

        for j in 0..n {
            let val = matrix.get(actual_row, j);
            if !val.is_zero() {
                if j < i {
                    // Below diagonal - goes to L
                    l_rows.push(i);
                    l_cols.push(j);
                    l_vals.push(val);
                } else {
                    // On or above diagonal - goes to U
                    u_rows.push(i);
                    u_cols.push(j);
                    u_vals.push(val);
                }
            }
        }
    }

    (l_rows, l_cols, l_vals, u_rows, u_cols, u_vals)
}

/// Extract lower triangular matrix
#[allow(dead_code)]
fn extract_lower_triangular<T>(
    matrix: &SparseWorkingMatrix<T>,
    n: usize,
) -> (Vec<usize>, Vec<usize>, Vec<T>)
where
    T: Float + Debug + Copy,
{
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        for j in 0..=i {
            let val = matrix.get(i, j);
            if !val.is_zero() {
                rows.push(i);
                cols.push(j);
                vals.push(val);
            }
        }
    }

    (rows, cols, vals)
}

/// Convert dense matrix to sparse
#[allow(dead_code)]
fn dense_to_sparse<T>(matrix: &Array2<T>) -> SparseResult<CsrArray<T>>
where
    T: Float + Debug + Copy,
{
    let (m, n) = matrix.dim();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..m {
        for j in 0..n {
            let val = matrix[[i, j]];
            if !val.is_zero() {
                rows.push(i);
                cols.push(j);
                vals.push(val);
            }
        }
    }

    CsrArray::from_triplets(&rows, &cols, &vals, (m, n), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;

    fn create_test_matrix() -> CsrArray<f64> {
        // Create a simple test matrix
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, 1.0, 1.0, 3.0, 2.0, 4.0];

        CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap()
    }

    fn create_spd_matrix() -> CsrArray<f64> {
        // Create a symmetric positive definite matrix
        let rows = vec![0, 1, 1, 2, 2, 2];
        let cols = vec![0, 0, 1, 0, 1, 2];
        let data = vec![4.0, 2.0, 5.0, 1.0, 3.0, 6.0];

        CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap()
    }

    #[test]
    fn test_lu_decomposition() {
        let matrix = create_test_matrix();
        let lu_result = lu_decomposition(&matrix, 0.1).unwrap();

        assert!(lu_result.success);
        assert_eq!(lu_result.l.shape(), (3, 3));
        assert_eq!(lu_result.u.shape(), (3, 3));
        assert_eq!(lu_result.p.len(), 3);
    }

    #[test]
    fn test_qr_decomposition() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 0, 1];
        let data = vec![1.0, 2.0, 3.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 2), false).unwrap();

        let qr_result = qr_decomposition(&matrix).unwrap();

        assert!(qr_result.success);
        assert_eq!(qr_result.q.shape(), (3, 2));
        assert_eq!(qr_result.r.shape(), (2, 2));
    }

    #[test]
    fn test_cholesky_decomposition() {
        let matrix = create_spd_matrix();
        let chol_result = cholesky_decomposition(&matrix).unwrap();

        assert!(chol_result.success);
        assert_eq!(chol_result.l.shape(), (3, 3));
    }

    #[test]
    fn test_incomplete_lu() {
        let matrix = create_test_matrix();
        let options = ILUOptions {
            drop_tol: 1e-6,
            ..Default::default()
        };

        let ilu_result = incomplete_lu(&matrix, Some(options)).unwrap();

        assert!(ilu_result.success);
        assert_eq!(ilu_result.l.shape(), (3, 3));
        assert_eq!(ilu_result.u.shape(), (3, 3));
    }

    #[test]
    fn test_incomplete_cholesky() {
        let matrix = create_spd_matrix();
        let options = ICOptions {
            drop_tol: 1e-6,
            ..Default::default()
        };

        let ic_result = incomplete_cholesky(&matrix, Some(options)).unwrap();

        assert!(ic_result.success);
        assert_eq!(ic_result.l.shape(), (3, 3));
    }

    #[test]
    fn test_sparse_working_matrix() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let vals = vec![1.0, 2.0, 3.0];

        let mut matrix = SparseWorkingMatrix::from_triplets(&rows, &cols, &vals, 3);

        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(1, 1), 2.0);
        assert_eq!(matrix.get(2, 2), 3.0);
        assert_eq!(matrix.get(0, 1), 0.0);

        matrix.set(0, 1, 5.0);
        assert_eq!(matrix.get(0, 1), 5.0);

        matrix.set(0, 1, 0.0);
        assert_eq!(matrix.get(0, 1), 0.0);
        assert!(!matrix.has_entry(0, 1));
    }

    #[test]
    fn test_dense_to_sparse_conversion() {
        let dense = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 2.0, 3.0]).unwrap();
        let sparse = dense_to_sparse(&dense).unwrap();

        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.get(0, 0), 1.0);
        assert_eq!(sparse.get(0, 1), 0.0);
        assert_eq!(sparse.get(1, 0), 2.0);
        assert_eq!(sparse.get(1, 1), 3.0);
    }
}
