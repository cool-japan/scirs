//! Sparse tensor representations and operations in coordinate (COO) format.
//!
//! This module provides a generic, const-parameterised `SparseTensor<N>` for
//! N-dimensional tensors stored in coordinate format, together with:
//!
//! - Conversion from / to dense [`ndarray::ArrayD`]
//! - Sparse matrix-matrix product (SpMM) for 2-D sparse tensors
//! - Tensor-vector mode product for 3-D sparse tensors
//! - Non-zero ratio (sparsity statistics)
//!
//! ## Coordinate (COO) format
//!
//! Each non-zero element is represented by a [`SparseTensorEntry<N>`] that
//! stores both the multi-index and the value.  Duplicate indices are allowed
//! and their values are summed whenever an operation requires it; the
//! internal representation is kept in *unsorted, deduplicated* form by
//! [`SparseTensor::canonicalize`].

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Entry type
// ---------------------------------------------------------------------------

/// A single non-zero entry in an N-dimensional sparse tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseTensorEntry<const N: usize> {
    /// Multi-index of the entry.
    pub indices: [usize; N],
    /// Value of the entry.
    pub value: f64,
}

impl<const N: usize> SparseTensorEntry<N> {
    /// Create a new entry.
    pub fn new(indices: [usize; N], value: f64) -> Self {
        Self { indices, value }
    }
}

// ---------------------------------------------------------------------------
// SparseTensor
// ---------------------------------------------------------------------------

/// Sparse N-dimensional tensor in coordinate (COO) format.
///
/// The tensor stores only its non-zero entries together with their
/// multi-indices.  Entries with the same index can be present multiple times;
/// call [`SparseTensor::canonicalize`] to merge them.
///
/// # Type parameters
///
/// * `N` – Tensor order (number of dimensions), fixed at compile time.
#[derive(Debug, Clone)]
pub struct SparseTensor<const N: usize> {
    /// Shape of the tensor (length-N array of dimension sizes).
    pub shape: [usize; N],
    /// Non-zero entries.
    pub entries: Vec<SparseTensorEntry<N>>,
}

impl<const N: usize> SparseTensor<N> {
    /// Create an empty sparse tensor with the given shape.
    pub fn new(shape: [usize; N]) -> Self {
        Self {
            shape,
            entries: Vec::new(),
        }
    }

    /// Add an entry to the tensor (does not check for duplicates).
    pub fn push(&mut self, indices: [usize; N], value: f64) {
        if value != 0.0 {
            self.entries.push(SparseTensorEntry::new(indices, value));
        }
    }

    /// Number of stored (non-zero) entries.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Total number of elements (product of all dimensions).
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Fraction of non-zero elements: `nnz / total_elements`.
    ///
    /// Returns `0.0` if the tensor has zero total elements.
    pub fn nnz_ratio(&self) -> f64 {
        let total = self.size();
        if total == 0 {
            0.0
        } else {
            self.entries.len() as f64 / total as f64
        }
    }

    /// Sort entries lexicographically by index and merge duplicates by summing
    /// their values.
    ///
    /// After calling this function, entries are in sorted order with no
    /// duplicate indices.
    pub fn canonicalize(&mut self) {
        // Sort by indices (lexicographic)
        self.entries.sort_by_key(|a| a.indices);

        // Merge duplicates
        let mut merged: Vec<SparseTensorEntry<N>> = Vec::with_capacity(self.entries.len());
        for entry in self.entries.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.indices == entry.indices {
                    last.value += entry.value;
                    continue;
                }
            }
            merged.push(entry);
        }
        // Remove exact zeros that may have appeared after merging
        merged.retain(|e| e.value != 0.0);
        self.entries = merged;
    }
}

// ---------------------------------------------------------------------------
// Conversion from dense
// ---------------------------------------------------------------------------

/// Build a [`SparseTensor<N>`] from a dense [`ArrayD<f64>`].
///
/// Elements whose absolute value is ≤ `threshold` are treated as zero and
/// omitted from the sparse representation.
///
/// # Arguments
///
/// * `tensor`    – Dense N-dimensional array.
/// * `threshold` – Elements with `|value| <= threshold` are stored as zero.
///
/// # Errors
///
/// Returns [`LinalgError::DimensionError`] if the array's number of dimensions
/// does not match `N`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{ArrayD, IxDyn};
/// use scirs2_linalg::sparse_tensor::{from_dense, SparseTensor};
///
/// let data = ArrayD::<f64>::from_shape_vec(
///     IxDyn(&[2, 3]),
///     vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
/// ).expect("shape error");
/// let sparse: SparseTensor<2> = from_dense(&data, 0.0).expect("conversion failed");
/// assert_eq!(sparse.nnz(), 3);
/// ```
pub fn from_dense<const N: usize>(
    tensor: &ArrayD<f64>,
    threshold: f64,
) -> LinalgResult<SparseTensor<N>> {
    if tensor.ndim() != N {
        return Err(LinalgError::DimensionError(format!(
            "from_dense: array has {} dimensions, expected {}",
            tensor.ndim(),
            N
        )));
    }

    let mut shape = [0usize; N];
    for (i, &s) in tensor.shape().iter().enumerate() {
        shape[i] = s;
    }

    let mut sparse = SparseTensor::new(shape);

    for (linear_idx, &value) in tensor.iter().enumerate() {
        if value.abs() <= threshold {
            continue;
        }
        // Convert linear index to multi-index (row-major)
        let mut indices = [0usize; N];
        let mut remaining = linear_idx;
        for dim in (0..N).rev() {
            indices[dim] = remaining % shape[dim];
            remaining /= shape[dim];
        }
        sparse.entries.push(SparseTensorEntry::new(indices, value));
    }

    Ok(sparse)
}

// ---------------------------------------------------------------------------
// Conversion to dense
// ---------------------------------------------------------------------------

/// Reconstruct a dense [`ArrayD<f64>`] from a [`SparseTensor<N>`].
///
/// Duplicate entries are summed.  Elements not present in the sparse tensor
/// are zero in the output.
///
/// # Arguments
///
/// * `sparse` – Sparse tensor in COO format.
/// * `shape`  – Output shape (must match `sparse.shape`).
///
/// # Errors
///
/// Returns [`LinalgError::DimensionError`] if `shape` has the wrong length.
/// Returns [`LinalgError::IndexError`] if any entry's index is out of bounds.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{ArrayD, IxDyn};
/// use scirs2_linalg::sparse_tensor::{from_dense, to_dense, SparseTensor};
///
/// let data = ArrayD::<f64>::from_shape_vec(
///     IxDyn(&[2, 3]),
///     vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
/// ).expect("shape error");
/// let sparse: SparseTensor<2> = from_dense(&data, 0.0).expect("from_dense failed");
/// let dense = to_dense(&sparse, &[2, 3]).expect("to_dense failed");
/// assert_eq!(dense, data);
/// ```
pub fn to_dense<const N: usize>(
    sparse: &SparseTensor<N>,
    shape: &[usize],
) -> LinalgResult<ArrayD<f64>> {
    if shape.len() != N {
        return Err(LinalgError::DimensionError(format!(
            "to_dense: shape slice has {} elements, expected {}",
            shape.len(),
            N
        )));
    }

    let mut out = ArrayD::<f64>::zeros(IxDyn(shape));

    for entry in &sparse.entries {
        // Build the IxDyn index
        let idx = IxDyn(entry.indices.as_slice());
        // Bounds check
        for dim in 0..N {
            if entry.indices[dim] >= shape[dim] {
                return Err(LinalgError::IndexError(format!(
                    "to_dense: entry index {} in dimension {} is out of bounds (size {})",
                    entry.indices[dim], dim, shape[dim]
                )));
            }
        }
        out[idx] += entry.value;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Sparse matrix-matrix product (SpMM) for 2-D sparse tensors
// ---------------------------------------------------------------------------

/// Sparse matrix-matrix multiplication `C = A * B` for 2-D sparse tensors.
///
/// Computes the matrix product of two sparse matrices in COO format.
/// The result is returned as a new sparse tensor in COO format (not yet
/// canonicalized; call [`SparseTensor::canonicalize`] if needed).
///
/// # Arguments
///
/// * `a` – Left-hand sparse matrix of shape `[m, k]`.
/// * `b` – Right-hand sparse matrix of shape `[k, n]`.
///
/// # Errors
///
/// Returns [`LinalgError::DimensionError`] if the inner dimensions do not match.
///
/// # Examples
///
/// ```
/// use scirs2_linalg::sparse_tensor::{SparseTensor, sparse_tensor_product};
///
/// // Identity matrices
/// let mut eye2: SparseTensor<2> = SparseTensor::new([2, 2]);
/// eye2.push([0, 0], 1.0);
/// eye2.push([1, 1], 1.0);
///
/// let mut b: SparseTensor<2> = SparseTensor::new([2, 2]);
/// b.push([0, 0], 3.0);
/// b.push([1, 1], 4.0);
///
/// let c = sparse_tensor_product(&eye2, &b).expect("SpMM failed");
/// assert_eq!(c.nnz(), 2);
/// ```
pub fn sparse_tensor_product(
    a: &SparseTensor<2>,
    b: &SparseTensor<2>,
) -> LinalgResult<SparseTensor<2>> {
    let m = a.shape[0];
    let k_a = a.shape[1];
    let k_b = b.shape[0];
    let n = b.shape[1];

    if k_a != k_b {
        return Err(LinalgError::DimensionError(format!(
            "sparse_tensor_product: A has {} columns but B has {} rows",
            k_a, k_b
        )));
    }

    // Group B entries by row index for fast lookup
    // b_by_row[row] = list of (col, value)
    let mut b_by_row: Vec<Vec<(usize, f64)>> = vec![Vec::new(); k_b];
    for entry in &b.entries {
        b_by_row[entry.indices[0]].push((entry.indices[1], entry.value));
    }

    let mut c = SparseTensor::new([m, n]);

    for a_entry in &a.entries {
        let row_a = a_entry.indices[0];
        let col_a = a_entry.indices[1]; // = shared dimension k
        let val_a = a_entry.value;

        for &(col_b, val_b) in &b_by_row[col_a] {
            c.entries
                .push(SparseTensorEntry::new([row_a, col_b], val_a * val_b));
        }
    }

    // Merge duplicate entries
    c.canonicalize();
    Ok(c)
}

// ---------------------------------------------------------------------------
// Tensor-vector mode product (for 3-D sparse tensors)
// ---------------------------------------------------------------------------

/// Multiply a 3-D sparse tensor by a dense vector along a given mode.
///
/// The n-mode product of an order-3 tensor `T` of shape `[I, J, K]` with a
/// vector `v` of length `dim_mode` contracts the specified mode:
///
/// - `mode = 0`: result shape `[J, K]`, `C[j,k] = Σ_i T[i,j,k] * v[i]`
/// - `mode = 1`: result shape `[I, K]`, `C[i,k] = Σ_j T[i,j,k] * v[j]`
/// - `mode = 2`: result shape `[I, J]`, `C[i,j] = Σ_k T[i,j,k] * v[k]`
///
/// # Arguments
///
/// * `t`    – Sparse 3-D tensor of shape `[I, J, K]`.
/// * `v`    – Dense vector of length equal to `t.shape[mode]`.
/// * `mode` – Mode index (0, 1, or 2).
///
/// # Returns
///
/// A [`SparseTensor<2>`] representing the contracted matrix (shape depends on
/// `mode`).
///
/// # Errors
///
/// Returns [`LinalgError::ValueError`] if `mode >= 3`.
/// Returns [`LinalgError::DimensionError`] if `v.len() != t.shape[mode]`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::sparse_tensor::{SparseTensor, tensor_times_vector};
///
/// // 2×2×2 identity-like tensor: T[i,i,i] = 1
/// let mut t: SparseTensor<3> = SparseTensor::new([2, 2, 2]);
/// t.push([0, 0, 0], 1.0);
/// t.push([1, 1, 1], 1.0);
///
/// let v = array![3.0_f64, 5.0];
/// let c = tensor_times_vector(&t, &v, 2).expect("TTV failed");
/// // C[0,0] = T[0,0,0] * v[0] = 3.0
/// // C[1,1] = T[1,1,1] * v[1] = 5.0
/// assert_eq!(c.nnz(), 2);
/// ```
pub fn tensor_times_vector(
    t: &SparseTensor<3>,
    v: &Array1<f64>,
    mode: usize,
) -> LinalgResult<SparseTensor<2>> {
    if mode >= 3 {
        return Err(LinalgError::ValueError(format!(
            "tensor_times_vector: mode must be 0, 1, or 2, got {}",
            mode
        )));
    }
    if v.len() != t.shape[mode] {
        return Err(LinalgError::DimensionError(format!(
            "tensor_times_vector: vector length {} does not match tensor dimension {} in mode {}",
            v.len(),
            t.shape[mode],
            mode
        )));
    }

    // Compute output shape
    let out_shape: [usize; 2] = match mode {
        0 => [t.shape[1], t.shape[2]],
        1 => [t.shape[0], t.shape[2]],
        2 => [t.shape[0], t.shape[1]],
        _ => unreachable!(),
    };

    let mut result: SparseTensor<2> = SparseTensor::new(out_shape);

    for entry in &t.entries {
        let [i, j, k] = entry.indices;
        let contracted_idx = match mode {
            0 => i,
            1 => j,
            2 => k,
            _ => unreachable!(),
        };
        let v_val = v[contracted_idx];
        let product = entry.value * v_val;
        if product == 0.0 {
            continue;
        }
        let out_idx: [usize; 2] = match mode {
            0 => [j, k],
            1 => [i, k],
            2 => [i, j],
            _ => unreachable!(),
        };
        result
            .entries
            .push(SparseTensorEntry::new(out_idx, product));
    }

    result.canonicalize();
    Ok(result)
}

// ---------------------------------------------------------------------------
// nnz_ratio convenience function
// ---------------------------------------------------------------------------

/// Return the fraction of non-zero entries in a sparse tensor.
///
/// This is a free-function wrapper around [`SparseTensor::nnz_ratio`].
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{ArrayD, IxDyn};
/// use scirs2_linalg::sparse_tensor::{from_dense, nnz_ratio};
///
/// let data = ArrayD::<f64>::from_shape_vec(
///     IxDyn(&[2, 4]),
///     vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
/// ).expect("shape error");
/// let sparse = from_dense::<2>(&data, 0.0).expect("from_dense failed");
/// let ratio = nnz_ratio(&sparse);
/// assert!((ratio - 0.25).abs() < 1e-15, "expected 0.25, got {}", ratio);
/// ```
pub fn nnz_ratio<const N: usize>(sparse: &SparseTensor<N>) -> f64 {
    sparse.nnz_ratio()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, ArrayD, IxDyn};

    // ---- from_dense / to_dense ----

    #[test]
    fn test_from_dense_basic() {
        let data =
            ArrayD::<f64>::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
                .expect("shape error");
        let sparse: SparseTensor<2> = from_dense(&data, 0.0).expect("from_dense failed");
        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.shape, [2, 3]);
    }

    #[test]
    fn test_from_dense_threshold() {
        let data =
            ArrayD::<f64>::from_shape_vec(IxDyn(&[3]), vec![1.0, 0.1, 1e-10]).expect("shape error");
        let sparse: SparseTensor<1> = from_dense(&data, 1e-9).expect("from_dense failed");
        // 1e-10 <= 1e-9, so only 2 entries should survive
        assert_eq!(sparse.nnz(), 2);
    }

    #[test]
    fn test_to_dense_roundtrip() {
        let data =
            ArrayD::<f64>::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
                .expect("shape error");
        let sparse: SparseTensor<2> = from_dense(&data, 0.0).expect("from_dense failed");
        let dense = to_dense(&sparse, &[2, 3]).expect("to_dense failed");
        assert_eq!(dense, data);
    }

    #[test]
    fn test_to_dense_duplicates_summed() {
        let mut sparse: SparseTensor<2> = SparseTensor::new([2, 2]);
        sparse.push([0, 0], 1.0);
        sparse.push([0, 0], 2.0); // duplicate; should sum to 3
        sparse.push([1, 1], 4.0);
        let dense = to_dense(&sparse, &[2, 2]).expect("to_dense failed");
        assert!((dense[[0, 0]] - 3.0).abs() < 1e-15);
        assert!((dense[[1, 1]] - 4.0).abs() < 1e-15);
        assert_eq!(dense[[0, 1]], 0.0);
        assert_eq!(dense[[1, 0]], 0.0);
    }

    #[test]
    fn test_from_dense_wrong_dims() {
        let data =
            ArrayD::<f64>::from_shape_vec(IxDyn(&[2, 3, 4]), vec![0.0; 24]).expect("shape error");
        // Trying to create SparseTensor<2> from a 3-D array should fail
        let result = from_dense::<2>(&data, 0.0);
        assert!(result.is_err());
    }

    // ---- nnz_ratio ----

    #[test]
    fn test_nnz_ratio() {
        let data = ArrayD::<f64>::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        )
        .expect("shape error");
        let sparse: SparseTensor<2> = from_dense(&data, 0.0).expect("from_dense failed");
        let r = nnz_ratio(&sparse);
        assert!((r - 0.25).abs() < 1e-15, "expected 0.25, got {}", r);
    }

    #[test]
    fn test_nnz_ratio_empty() {
        let sparse: SparseTensor<2> = SparseTensor::new([0, 0]);
        assert_eq!(nnz_ratio(&sparse), 0.0);
    }

    // ---- sparse_tensor_product ----

    #[test]
    fn test_spmm_identity() {
        // I * B = B
        let mut eye: SparseTensor<2> = SparseTensor::new([3, 3]);
        eye.push([0, 0], 1.0);
        eye.push([1, 1], 1.0);
        eye.push([2, 2], 1.0);

        let mut b: SparseTensor<2> = SparseTensor::new([3, 2]);
        b.push([0, 0], 1.0);
        b.push([1, 1], 2.0);
        b.push([2, 0], 3.0);

        let c = sparse_tensor_product(&eye, &b).expect("SpMM failed");
        let dense_c = to_dense(&c, &[3, 2]).expect("to_dense failed");
        let dense_b = to_dense(&b, &[3, 2]).expect("to_dense b failed");
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (dense_c[[i, j]] - dense_b[[i, j]]).abs() < 1e-12,
                    "I*B != B at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_spmm_general() {
        // A = [[1,2],[3,4]]  B = [[5,6],[7,8]]  C = A*B
        let mut a: SparseTensor<2> = SparseTensor::new([2, 2]);
        a.push([0, 0], 1.0);
        a.push([0, 1], 2.0);
        a.push([1, 0], 3.0);
        a.push([1, 1], 4.0);

        let mut b: SparseTensor<2> = SparseTensor::new([2, 2]);
        b.push([0, 0], 5.0);
        b.push([0, 1], 6.0);
        b.push([1, 0], 7.0);
        b.push([1, 1], 8.0);

        let c = sparse_tensor_product(&a, &b).expect("SpMM failed");
        let dense = to_dense(&c, &[2, 2]).expect("to_dense failed");
        // A*B = [[19, 22], [43, 50]]
        assert!((dense[[0, 0]] - 19.0).abs() < 1e-12);
        assert!((dense[[0, 1]] - 22.0).abs() < 1e-12);
        assert!((dense[[1, 0]] - 43.0).abs() < 1e-12);
        assert!((dense[[1, 1]] - 50.0).abs() < 1e-12);
    }

    #[test]
    fn test_spmm_dimension_mismatch() {
        let a: SparseTensor<2> = SparseTensor::new([2, 3]);
        let b: SparseTensor<2> = SparseTensor::new([4, 2]); // inner dim mismatch
        assert!(sparse_tensor_product(&a, &b).is_err());
    }

    // ---- tensor_times_vector ----

    #[test]
    fn test_ttv_mode0() {
        // T[i,j,k] = delta_{ijk} (identity-like)
        let mut t: SparseTensor<3> = SparseTensor::new([2, 2, 2]);
        t.push([0, 0, 0], 1.0);
        t.push([1, 1, 1], 1.0);

        let v = array![2.0_f64, 3.0];
        // mode=0: C[j,k] = sum_i T[i,j,k]*v[i]
        // C[0,0] = T[0,0,0]*v[0] = 2
        // C[1,1] = T[1,1,1]*v[1] = 3
        let c = tensor_times_vector(&t, &v, 0).expect("TTV failed");
        assert_eq!(c.shape, [2, 2]);
        let dense = to_dense(&c, &[2, 2]).expect("to_dense failed");
        assert!((dense[[0, 0]] - 2.0).abs() < 1e-12);
        assert!((dense[[1, 1]] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_ttv_mode2() {
        let mut t: SparseTensor<3> = SparseTensor::new([2, 2, 2]);
        t.push([0, 0, 0], 1.0);
        t.push([0, 1, 1], 2.0);
        t.push([1, 0, 0], 3.0);
        t.push([1, 1, 1], 4.0);

        let v = array![5.0_f64, 7.0];
        // mode=2: C[i,j] = sum_k T[i,j,k]*v[k]
        // C[0,0] = 1*5 = 5
        // C[0,1] = 2*7 = 14
        // C[1,0] = 3*5 = 15
        // C[1,1] = 4*7 = 28
        let c = tensor_times_vector(&t, &v, 2).expect("TTV failed");
        let dense = to_dense(&c, &[2, 2]).expect("to_dense failed");
        assert!((dense[[0, 0]] - 5.0).abs() < 1e-12);
        assert!((dense[[0, 1]] - 14.0).abs() < 1e-12);
        assert!((dense[[1, 0]] - 15.0).abs() < 1e-12);
        assert!((dense[[1, 1]] - 28.0).abs() < 1e-12);
    }

    #[test]
    fn test_ttv_wrong_mode() {
        let t: SparseTensor<3> = SparseTensor::new([2, 2, 2]);
        let v = array![1.0_f64];
        assert!(tensor_times_vector(&t, &v, 3).is_err());
    }

    #[test]
    fn test_ttv_dim_mismatch() {
        let t: SparseTensor<3> = SparseTensor::new([2, 3, 4]);
        let v = array![1.0_f64, 2.0, 3.0]; // length 3, but mode 0 expects 2
        assert!(tensor_times_vector(&t, &v, 0).is_err());
    }

    // ---- canonicalize ----

    #[test]
    fn test_canonicalize_merges_duplicates() {
        let mut sparse: SparseTensor<2> = SparseTensor::new([3, 3]);
        sparse.push([0, 0], 1.0);
        sparse.push([0, 0], 2.0);
        sparse.push([1, 2], 3.0);
        sparse.canonicalize();
        assert_eq!(sparse.nnz(), 2);
        let e00 = sparse.entries.iter().find(|e| e.indices == [0, 0]);
        assert!(e00.is_some());
        assert!((e00.expect("missing").value - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_canonicalize_removes_zeros() {
        let mut sparse: SparseTensor<1> = SparseTensor::new([4]);
        sparse.entries.push(SparseTensorEntry::new([0], 1.0));
        sparse.entries.push(SparseTensorEntry::new([0], -1.0)); // sum = 0
        sparse.entries.push(SparseTensorEntry::new([2], 5.0));
        sparse.canonicalize();
        assert_eq!(sparse.nnz(), 1);
        assert_eq!(sparse.entries[0].indices, [2]);
    }

    // ---- size / nnz ----

    #[test]
    fn test_size_and_nnz() {
        let data =
            ArrayD::<f64>::from_shape_vec(IxDyn(&[3, 4, 5]), (0..60).map(|x| x as f64).collect())
                .expect("shape error");
        let sparse: SparseTensor<3> = from_dense(&data, 0.5).expect("from_dense failed");
        assert_eq!(sparse.size(), 60);
        // All values 1..59 plus 0 is filtered; 1..59 remain = 59
        assert_eq!(sparse.nnz(), 59);
    }
}
