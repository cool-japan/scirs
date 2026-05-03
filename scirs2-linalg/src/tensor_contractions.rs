//! Tensor contractions, Einstein summation, and related algebraic products.
//!
//! This module provides concrete, ergonomic interfaces for common tensor operations
//! on `Array2` / `Array3` objects (f64), complementing the generic dynamic-rank
//! interfaces in `tensor_contraction/`.
//!
//! ## Operations
//!
//! * [`einsum_2d`] – Einstein summation on two 2-D matrices.
//! * [`einsum_3d`] – Einstein summation on two 3-D arrays.
//! * [`tensor_mode_product`] – n-mode product of a 3-D tensor with a matrix.
//! * [`unfold_tensor`] – tensor matricization (unfolding) for a 3-D array.
//! * [`fold_tensor`] – inverse of `unfold_tensor`.
//! * [`khatri_rao`] – Khatri-Rao (column-wise Kronecker) product.
//! * [`hadamard`] – element-wise (Hadamard) product.
//! * [`kronecker`] – Kronecker product.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse an einsum subscript string of the form `"ab,bc->ac"` and return:
/// `(lhs_a_indices, lhs_b_indices, output_indices)`.
fn parse_einsum_str(subscripts: &str) -> LinalgResult<(Vec<char>, Vec<char>, Vec<char>)> {
    let parts: Vec<&str> = subscripts.split("->").collect();
    if parts.len() != 2 {
        return Err(LinalgError::ValueError(format!(
            "Einsum string must contain exactly one '->'; got: {subscripts}"
        )));
    }
    let inputs: Vec<&str> = parts[0].split(',').collect();
    if inputs.len() != 2 {
        return Err(LinalgError::ValueError(format!(
            "Einsum string must have exactly two input operands; got: {subscripts}"
        )));
    }
    let a_idx: Vec<char> = inputs[0].trim().chars().collect();
    let b_idx: Vec<char> = inputs[1].trim().chars().collect();
    let out_idx: Vec<char> = parts[1].trim().chars().collect();
    Ok((a_idx, b_idx, out_idx))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute Einstein summation on two 2-D matrices (f64).
///
/// The subscript string follows NumPy einsum convention, e.g.
/// `"ij,jk->ik"` for matrix multiplication or `"ij,ij->ij"` for element-wise product.
///
/// # Errors
///
/// Returns `LinalgError::ValueError` if the subscript string is malformed,
/// `LinalgError::ShapeError` if tensor dimensions are inconsistent.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::tensor_contractions::einsum_2d;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
/// // Matrix multiplication ij,jk->ik
/// let c = einsum_2d("ij,jk->ik", &a, &b).expect("valid input");
/// assert_eq!(c[[0, 0]], 1.0 * 5.0 + 2.0 * 7.0); // 19
/// assert_eq!(c[[0, 1]], 1.0 * 6.0 + 2.0 * 8.0); // 22
/// ```
pub fn einsum_2d(subscripts: &str, a: &Array2<f64>, b: &Array2<f64>) -> LinalgResult<Array2<f64>> {
    einsum_2d_view(subscripts, &a.view(), &b.view())
}

/// View-based variant of [`einsum_2d`].
pub fn einsum_2d_view(
    subscripts: &str,
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
) -> LinalgResult<Array2<f64>> {
    let (a_idx, b_idx, out_idx) = parse_einsum_str(subscripts)?;

    // Validate index counts match tensor ranks
    if a_idx.len() != 2 {
        return Err(LinalgError::ValueError(format!(
            "First operand subscript must have 2 indices for a 2D array; got {}",
            a_idx.len()
        )));
    }
    if b_idx.len() != 2 {
        return Err(LinalgError::ValueError(format!(
            "Second operand subscript must have 2 indices for a 2D array; got {}",
            b_idx.len()
        )));
    }
    if out_idx.len() != 2 {
        return Err(LinalgError::ValueError(format!(
            "Output subscript must have 2 indices for a 2D result; got {}",
            out_idx.len()
        )));
    }

    // Build dimension map
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    let a_shape = a.shape();
    let b_shape = b.shape();

    let insert_dim = |map: &mut HashMap<char, usize>, key: char, dim: usize| -> LinalgResult<()> {
        if let Some(&existing) = map.get(&key) {
            if existing != dim {
                return Err(LinalgError::ShapeError(format!(
                    "Inconsistent dimension for index '{key}': {existing} vs {dim}"
                )));
            }
        } else {
            map.insert(key, dim);
        }
        Ok(())
    };

    insert_dim(&mut dim_map, a_idx[0], a_shape[0])?;
    insert_dim(&mut dim_map, a_idx[1], a_shape[1])?;
    insert_dim(&mut dim_map, b_idx[0], b_shape[0])?;
    insert_dim(&mut dim_map, b_idx[1], b_shape[1])?;

    // Determine output dimensions
    let out0 = *dim_map.get(&out_idx[0]).ok_or_else(|| {
        LinalgError::ValueError(format!("Output index '{}' not in inputs", out_idx[0]))
    })?;
    let out1 = *dim_map.get(&out_idx[1]).ok_or_else(|| {
        LinalgError::ValueError(format!("Output index '{}' not in inputs", out_idx[1]))
    })?;

    // Identify contracted indices (appear in inputs but not in output)
    let all_input_indices: Vec<char> = a_idx.iter().chain(b_idx.iter()).copied().collect();
    let contracted: Vec<char> = all_input_indices
        .iter()
        .copied()
        .filter(|c| !out_idx.contains(c))
        .collect::<std::collections::HashSet<char>>()
        .into_iter()
        .collect();

    let mut result = Array2::<f64>::zeros((out0, out1));

    // Enumerate over output indices
    for i in 0..out0 {
        for j in 0..out1 {
            let mut idx_map: HashMap<char, usize> = HashMap::new();
            idx_map.insert(out_idx[0], i);
            idx_map.insert(out_idx[1], j);

            // Sum over contracted indices
            let sum = contract_indices_2d(a, b, &a_idx, &b_idx, &contracted, &dim_map, idx_map, 0);
            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

/// Recursive helper: sums over contracted indices for einsum_2d.
fn contract_indices_2d(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    a_idx: &[char],
    b_idx: &[char],
    contracted: &[char],
    dim_map: &HashMap<char, usize>,
    idx_map: HashMap<char, usize>,
    depth: usize,
) -> f64 {
    if depth == contracted.len() {
        // Evaluate a[i,j] * b[k,l] at current index assignment
        let ai = *idx_map.get(&a_idx[0]).unwrap_or(&0);
        let aj = *idx_map.get(&a_idx[1]).unwrap_or(&0);
        let bi = *idx_map.get(&b_idx[0]).unwrap_or(&0);
        let bj = *idx_map.get(&b_idx[1]).unwrap_or(&0);
        return a[[ai, aj]] * b[[bi, bj]];
    }
    let c = contracted[depth];
    let dim = *dim_map.get(&c).unwrap_or(&0);
    let mut total = 0.0_f64;
    for k in 0..dim {
        let mut new_map = idx_map.clone();
        new_map.insert(c, k);
        total += contract_indices_2d(a, b, a_idx, b_idx, contracted, dim_map, new_map, depth + 1);
    }
    total
}

/// Compute Einstein summation on two 3-D arrays (f64).
///
/// The subscript string follows NumPy einsum convention, e.g.
/// `"ijk,jkl->il"`.  The output must have exactly 3 free indices.
///
/// # Errors
///
/// Returns `LinalgError::ValueError` for malformed subscripts, or
/// `LinalgError::ShapeError` for incompatible shapes.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array3, array};
/// use scirs2_linalg::tensor_contractions::einsum_3d;
///
/// // Element-wise multiply then keep all dims: "ijk,ijk->ijk"
/// let a = Array3::<f64>::ones((2, 3, 4));
/// let b = Array3::<f64>::from_elem((2, 3, 4), 2.0);
/// let c = einsum_3d("ijk,ijk->ijk", &a, &b).expect("valid input");
/// assert_eq!(c[[0, 0, 0]], 2.0);
/// ```
pub fn einsum_3d(subscripts: &str, a: &Array3<f64>, b: &Array3<f64>) -> LinalgResult<Array3<f64>> {
    einsum_3d_view(subscripts, &a.view(), &b.view())
}

/// View-based variant of [`einsum_3d`].
pub fn einsum_3d_view(
    subscripts: &str,
    a: &ArrayView3<f64>,
    b: &ArrayView3<f64>,
) -> LinalgResult<Array3<f64>> {
    let (a_idx, b_idx, out_idx) = parse_einsum_str(subscripts)?;

    if a_idx.len() != 3 {
        return Err(LinalgError::ValueError(format!(
            "First operand subscript must have 3 indices; got {}",
            a_idx.len()
        )));
    }
    if b_idx.len() != 3 {
        return Err(LinalgError::ValueError(format!(
            "Second operand subscript must have 3 indices; got {}",
            b_idx.len()
        )));
    }
    if out_idx.len() != 3 {
        return Err(LinalgError::ValueError(format!(
            "Output subscript must have 3 indices for a 3D result; got {}",
            out_idx.len()
        )));
    }

    let mut dim_map: HashMap<char, usize> = HashMap::new();
    let a_shape = a.shape();
    let b_shape = b.shape();

    let insert_dim = |map: &mut HashMap<char, usize>, key: char, dim: usize| -> LinalgResult<()> {
        if let Some(&existing) = map.get(&key) {
            if existing != dim {
                return Err(LinalgError::ShapeError(format!(
                    "Inconsistent dimension for index '{key}': {existing} vs {dim}"
                )));
            }
        } else {
            map.insert(key, dim);
        }
        Ok(())
    };

    for (i, &c) in a_idx.iter().enumerate() {
        insert_dim(&mut dim_map, c, a_shape[i])?;
    }
    for (i, &c) in b_idx.iter().enumerate() {
        insert_dim(&mut dim_map, c, b_shape[i])?;
    }

    let out0 = *dim_map.get(&out_idx[0]).ok_or_else(|| {
        LinalgError::ValueError(format!("Output index '{}' not in inputs", out_idx[0]))
    })?;
    let out1 = *dim_map.get(&out_idx[1]).ok_or_else(|| {
        LinalgError::ValueError(format!("Output index '{}' not in inputs", out_idx[1]))
    })?;
    let out2 = *dim_map.get(&out_idx[2]).ok_or_else(|| {
        LinalgError::ValueError(format!("Output index '{}' not in inputs", out_idx[2]))
    })?;

    let all_input: Vec<char> = a_idx.iter().chain(b_idx.iter()).copied().collect();
    let contracted: Vec<char> = all_input
        .iter()
        .copied()
        .filter(|c| !out_idx.contains(c))
        .collect::<std::collections::HashSet<char>>()
        .into_iter()
        .collect();

    let mut result = Array3::<f64>::zeros((out0, out1, out2));

    for i in 0..out0 {
        for j in 0..out1 {
            for k in 0..out2 {
                let mut idx_map: HashMap<char, usize> = HashMap::new();
                idx_map.insert(out_idx[0], i);
                idx_map.insert(out_idx[1], j);
                idx_map.insert(out_idx[2], k);
                let sum =
                    contract_indices_3d(a, b, &a_idx, &b_idx, &contracted, &dim_map, idx_map, 0);
                result[[i, j, k]] = sum;
            }
        }
    }

    Ok(result)
}

/// Recursive helper: sums over contracted indices for einsum_3d.
fn contract_indices_3d(
    a: &ArrayView3<f64>,
    b: &ArrayView3<f64>,
    a_idx: &[char],
    b_idx: &[char],
    contracted: &[char],
    dim_map: &HashMap<char, usize>,
    idx_map: HashMap<char, usize>,
    depth: usize,
) -> f64 {
    if depth == contracted.len() {
        let ai = *idx_map.get(&a_idx[0]).unwrap_or(&0);
        let aj = *idx_map.get(&a_idx[1]).unwrap_or(&0);
        let ak = *idx_map.get(&a_idx[2]).unwrap_or(&0);
        let bi = *idx_map.get(&b_idx[0]).unwrap_or(&0);
        let bj = *idx_map.get(&b_idx[1]).unwrap_or(&0);
        let bk = *idx_map.get(&b_idx[2]).unwrap_or(&0);
        return a[[ai, aj, ak]] * b[[bi, bj, bk]];
    }
    let c = contracted[depth];
    let dim = *dim_map.get(&c).unwrap_or(&0);
    let mut total = 0.0_f64;
    for k in 0..dim {
        let mut new_map = idx_map.clone();
        new_map.insert(c, k);
        total += contract_indices_3d(a, b, a_idx, b_idx, contracted, dim_map, new_map, depth + 1);
    }
    total
}

/// Compute the n-mode product of a 3-D tensor with a 2-D matrix.
///
/// The n-mode product of tensor `T` (shape `I0 × I1 × I2`) with matrix `M` (shape `J × I_mode`)
/// yields a tensor of shape `I0 × I1 × I2` with `I_mode` replaced by `J`.
///
/// # Arguments
///
/// * `tensor` – Input tensor of shape `(I0, I1, I2)`.
/// * `matrix` – Matrix of shape `(J, I_mode)`.
/// * `mode`   – Mode index (0, 1, or 2).
///
/// # Errors
///
/// Returns `LinalgError::ShapeError` if `matrix.ncols() != tensor.shape()[mode]`,
/// or `LinalgError::ValueError` if `mode >= 3`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array2, Array3};
/// use scirs2_linalg::tensor_contractions::tensor_mode_product;
///
/// // Tensor of shape (2, 3, 2)
/// let t = Array3::from_shape_fn((2, 3, 2), |(i, j, k)| (i + j + k) as f64);
/// // Identity on mode 0 (2 × 2 identity)
/// let eye2 = Array2::<f64>::eye(2);
/// let out = tensor_mode_product(&t, &eye2, 0).expect("valid input");
/// assert_eq!(out.shape(), &[2, 3, 2]);
/// ```
pub fn tensor_mode_product(
    tensor: &Array3<f64>,
    matrix: &Array2<f64>,
    mode: usize,
) -> LinalgResult<Array3<f64>> {
    tensor_mode_product_view(&tensor.view(), &matrix.view(), mode)
}

/// View-based variant of [`tensor_mode_product`].
pub fn tensor_mode_product_view(
    tensor: &ArrayView3<f64>,
    matrix: &ArrayView2<f64>,
    mode: usize,
) -> LinalgResult<Array3<f64>> {
    if mode >= 3 {
        return Err(LinalgError::ValueError(format!(
            "Mode {mode} is out of range for a 3-D tensor (modes are 0, 1, 2)"
        )));
    }
    let shape = tensor.shape();
    if matrix.ncols() != shape[mode] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix has {} columns but tensor mode {mode} has size {}",
            matrix.ncols(),
            shape[mode]
        )));
    }
    let j = matrix.nrows();
    let (i0, i1, i2) = (shape[0], shape[1], shape[2]);

    let (out_shape, result) = match mode {
        0 => {
            let mut result = Array3::<f64>::zeros((j, i1, i2));
            for r in 0..j {
                for a in 0..i0 {
                    for p in 0..i1 {
                        for q in 0..i2 {
                            result[[r, p, q]] += matrix[[r, a]] * tensor[[a, p, q]];
                        }
                    }
                }
            }
            ((j, i1, i2), result)
        }
        1 => {
            let mut result = Array3::<f64>::zeros((i0, j, i2));
            for r in 0..j {
                for b in 0..i1 {
                    for p in 0..i0 {
                        for q in 0..i2 {
                            result[[p, r, q]] += matrix[[r, b]] * tensor[[p, b, q]];
                        }
                    }
                }
            }
            ((i0, j, i2), result)
        }
        2 => {
            let mut result = Array3::<f64>::zeros((i0, i1, j));
            for r in 0..j {
                for c in 0..i2 {
                    for p in 0..i0 {
                        for q in 0..i1 {
                            result[[p, q, r]] += matrix[[r, c]] * tensor[[p, q, c]];
                        }
                    }
                }
            }
            ((i0, i1, j), result)
        }
        _ => unreachable!(),
    };
    let _ = out_shape; // shape already encoded in `result`
    Ok(result)
}

/// Matricize (unfold) a 3-D tensor along the given mode.
///
/// The mode-`n` unfolding of a tensor `T` of shape `(I0, I1, I2)` produces
/// a matrix of shape `(I_n, J)` where `J = prod(I_k for k != n)`.
///
/// The column ordering follows the convention used in most tensor-decomposition
/// literature (last index cycles fastest among the "other" modes when traversed
/// from the highest to lowest mode index).
///
/// # Errors
///
/// Returns `LinalgError::ValueError` if `mode >= 3`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::tensor_contractions::unfold_tensor;
///
/// let t = Array3::<f64>::zeros((2, 3, 4));
/// let m = unfold_tensor(&t, 0).expect("valid input");
/// assert_eq!(m.shape(), &[2, 12]);
/// let m1 = unfold_tensor(&t, 1).expect("valid input");
/// assert_eq!(m1.shape(), &[3, 8]);
/// let m2 = unfold_tensor(&t, 2).expect("valid input");
/// assert_eq!(m2.shape(), &[4, 6]);
/// ```
pub fn unfold_tensor(tensor: &Array3<f64>, mode: usize) -> LinalgResult<Array2<f64>> {
    unfold_tensor_view(&tensor.view(), mode)
}

/// View-based variant of [`unfold_tensor`].
pub fn unfold_tensor_view(tensor: &ArrayView3<f64>, mode: usize) -> LinalgResult<Array2<f64>> {
    if mode >= 3 {
        return Err(LinalgError::ValueError(format!(
            "Mode {mode} is out of range for a 3-D tensor (modes are 0, 1, 2)"
        )));
    }
    let shape = tensor.shape();
    let (i0, i1, i2) = (shape[0], shape[1], shape[2]);
    let mode_dim = shape[mode];
    let other_dims: usize = shape
        .iter()
        .enumerate()
        .filter(|(d, _)| *d != mode)
        .map(|(_, &s)| s)
        .product();

    let mut result = Array2::<f64>::zeros((mode_dim, other_dims));

    // For each element, compute (row, col) in the unfolded matrix.
    // Row  = index along `mode`.
    // Col  = combined index of all other modes, last-mode-index cycles fastest.
    for p in 0..i0 {
        for q in 0..i1 {
            for r in 0..i2 {
                let row = match mode {
                    0 => p,
                    1 => q,
                    2 => r,
                    _ => unreachable!(),
                };
                // Build col by iterating other dims in reverse order (highest first).
                let col = match mode {
                    0 => q * i2 + r, // other dims = [1, 2]
                    1 => p * i2 + r, // other dims = [0, 2]
                    2 => p * i1 + q, // other dims = [0, 1]
                    _ => unreachable!(),
                };
                result[[row, col]] = tensor[[p, q, r]];
            }
        }
    }

    Ok(result)
}

/// Fold a matrix back into a 3-D tensor (inverse of [`unfold_tensor`]).
///
/// # Arguments
///
/// * `matrix` – The unfolded matrix of shape `(I_mode, J)`.
/// * `shape`  – Target tensor shape `[I0, I1, I2]`.
/// * `mode`   – Mode that was unfolded (0, 1, or 2).
///
/// # Errors
///
/// Returns errors if `shape` does not have exactly 3 elements, `mode >= 3`,
/// or the matrix dimensions are inconsistent with `shape` and `mode`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::tensor_contractions::{unfold_tensor, fold_tensor};
///
/// let t = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as f64);
/// let m = unfold_tensor(&t, 1).expect("valid input");
/// let t2 = fold_tensor(&m, [2, 3, 4], 1).expect("valid input");
/// assert_eq!(t, t2);
/// ```
pub fn fold_tensor(
    matrix: &Array2<f64>,
    shape: [usize; 3],
    mode: usize,
) -> LinalgResult<Array3<f64>> {
    fold_tensor_view(&matrix.view(), shape, mode)
}

/// View-based variant of [`fold_tensor`].
pub fn fold_tensor_view(
    matrix: &ArrayView2<f64>,
    shape: [usize; 3],
    mode: usize,
) -> LinalgResult<Array3<f64>> {
    if mode >= 3 {
        return Err(LinalgError::ValueError(format!(
            "Mode {mode} is out of range for a 3-D tensor"
        )));
    }
    let (i0, i1, i2) = (shape[0], shape[1], shape[2]);
    let mode_dim = shape[mode];
    let other_dims: usize = shape
        .iter()
        .enumerate()
        .filter(|(d, _)| *d != mode)
        .map(|(_, &s)| s)
        .product();

    if matrix.nrows() != mode_dim {
        return Err(LinalgError::ShapeError(format!(
            "Matrix has {} rows but shape[{mode}] = {mode_dim}",
            matrix.nrows()
        )));
    }
    if matrix.ncols() != other_dims {
        return Err(LinalgError::ShapeError(format!(
            "Matrix has {} columns but product of other dims = {other_dims}",
            matrix.ncols()
        )));
    }

    let mut result = Array3::<f64>::zeros((i0, i1, i2));

    for p in 0..i0 {
        for q in 0..i1 {
            for r in 0..i2 {
                let row = match mode {
                    0 => p,
                    1 => q,
                    2 => r,
                    _ => unreachable!(),
                };
                let col = match mode {
                    0 => q * i2 + r,
                    1 => p * i2 + r,
                    2 => p * i1 + q,
                    _ => unreachable!(),
                };
                result[[p, q, r]] = matrix[[row, col]];
            }
        }
    }

    Ok(result)
}

/// Compute the Khatri-Rao (column-wise Kronecker) product of two matrices.
///
/// Given `A` of shape `(I, R)` and `B` of shape `(J, R)`, the Khatri-Rao product
/// is an `(I·J, R)` matrix whose `r`-th column is `kron(A[:,r], B[:,r])`.
///
/// # Errors
///
/// Returns `LinalgError::ShapeError` if the two matrices have different numbers
/// of columns.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::tensor_contractions::khatri_rao;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[5.0, 6.0], [7.0, 8.0]];
/// let kr = khatri_rao(&a, &b).expect("valid input");
/// assert_eq!(kr.shape(), &[4, 2]);
/// // Col 0: kron([1,3], [5,7]) = [5, 7, 15, 21]
/// assert_eq!(kr[[0, 0]], 5.0);
/// assert_eq!(kr[[1, 0]], 7.0);
/// assert_eq!(kr[[2, 0]], 15.0);
/// assert_eq!(kr[[3, 0]], 21.0);
/// ```
pub fn khatri_rao(a: &Array2<f64>, b: &Array2<f64>) -> LinalgResult<Array2<f64>> {
    khatri_rao_view(&a.view(), &b.view())
}

/// View-based variant of [`khatri_rao`].
pub fn khatri_rao_view(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let (ia, ra) = (a.nrows(), a.ncols());
    let (jb, rb) = (b.nrows(), b.ncols());
    if ra != rb {
        return Err(LinalgError::ShapeError(format!(
            "Khatri-Rao product requires equal column counts; A has {ra}, B has {rb}"
        )));
    }
    let r = ra;
    let mut result = Array2::<f64>::zeros((ia * jb, r));
    for col in 0..r {
        for i in 0..ia {
            for j in 0..jb {
                result[[i * jb + j, col]] = a[[i, col]] * b[[j, col]];
            }
        }
    }
    Ok(result)
}

/// Compute the Hadamard (element-wise) product of two matrices of the same shape.
///
/// This is equivalent to `a * b` element-wise, but is provided here for
/// completeness alongside the other tensor products.
///
/// # Errors
///
/// Returns `LinalgError::ShapeError` if the matrices have different shapes.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::tensor_contractions::hadamard;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[5.0, 6.0], [7.0, 8.0]];
/// let h = hadamard(&a, &b).expect("valid input");
/// assert_eq!(h[[0, 0]], 5.0);
/// assert_eq!(h[[1, 1]], 32.0);
/// ```
pub fn hadamard(a: &Array2<f64>, b: &Array2<f64>) -> LinalgResult<Array2<f64>> {
    hadamard_view(&a.view(), &b.view())
}

/// View-based variant of [`hadamard`].
pub fn hadamard_view(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    if a.shape() != b.shape() {
        return Err(LinalgError::ShapeError(format!(
            "Hadamard product requires equal shapes; A is {:?}, B is {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let (m, n) = (a.nrows(), a.ncols());
    let mut result = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = a[[i, j]] * b[[i, j]];
        }
    }
    Ok(result)
}

/// Compute the Kronecker product of two matrices.
///
/// Given `A` of shape `(m, n)` and `B` of shape `(p, q)`, the Kronecker product
/// is an `(m·p, n·q)` matrix.
///
/// The block structure is: `result[ i*p .. (i+1)*p, j*q .. (j+1)*q ] = A[i,j] * B`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::tensor_contractions::kronecker;
///
/// let a = array![[1.0, 0.0], [0.0, 1.0]]; // 2×2 identity
/// let b = array![[1.0, 2.0], [3.0, 4.0]];
/// let k = kronecker(&a, &b).expect("valid input");
/// assert_eq!(k.shape(), &[4, 4]);
/// // Top-left block: 1 * b
/// assert_eq!(k[[0, 0]], 1.0);
/// assert_eq!(k[[0, 1]], 2.0);
/// // Bottom-right block: 1 * b
/// assert_eq!(k[[2, 2]], 1.0);
/// assert_eq!(k[[3, 3]], 4.0);
/// ```
pub fn kronecker(a: &Array2<f64>, b: &Array2<f64>) -> LinalgResult<Array2<f64>> {
    kronecker_view(&a.view(), &b.view())
}

/// View-based variant of [`kronecker`].
pub fn kronecker_view(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let (m, n) = (a.nrows(), a.ncols());
    let (p, q) = (b.nrows(), b.ncols());
    let mut result = Array2::<f64>::zeros((m * p, n * q));
    for i in 0..m {
        for j in 0..n {
            let aij = a[[i, j]];
            for r in 0..p {
                for s in 0..q {
                    result[[i * p + r, j * q + s]] = aij * b[[r, s]];
                }
            }
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array3};

    // --- einsum_2d ---

    #[test]
    fn test_einsum_2d_matmul() {
        // ij,jk->ik = matrix multiplication
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2×3
        let b = array![[7.0_f64, 8.0], [9.0, 10.0], [11.0, 12.0]]; // 3×2
        let c = einsum_2d("ij,jk->ik", &a, &b).expect("einsum_2d ok");
        assert_eq!(c.shape(), &[2, 2]);
        // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        assert_abs_diff_eq!(c[[0, 0]], 58.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[0, 1]], 64.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 0]], 139.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 1]], 154.0, epsilon = 1e-10);
    }

    #[test]
    fn test_einsum_2d_elementwise() {
        // ij,ij->ij = element-wise product
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
        let c = einsum_2d("ij,ij->ij", &a, &b).expect("einsum_2d elem ok");
        assert_abs_diff_eq!(c[[0, 0]], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 1]], 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_einsum_2d_transpose() {
        // ij,ji->ij = a[i,j] * b[j,i]
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let c = einsum_2d("ij,ji->ij", &a, &b).expect("einsum_2d transpose ok");
        // c[i,j] = a[i,j] * b[j,i]
        assert_abs_diff_eq!(c[[0, 0]], 1.0 * 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[0, 1]], 2.0 * 3.0, epsilon = 1e-10); // a[0,1]*b[1,0]
        assert_abs_diff_eq!(c[[1, 0]], 3.0 * 2.0, epsilon = 1e-10); // a[1,0]*b[0,1]
    }

    #[test]
    fn test_einsum_2d_bad_subscript() {
        let a = array![[1.0_f64]];
        let b = array![[1.0_f64]];
        assert!(einsum_2d("no_arrow", &a, &b).is_err());
        assert!(einsum_2d("ij->ij", &a, &b).is_err()); // only one operand
    }

    // --- einsum_3d ---

    #[test]
    fn test_einsum_3d_elementwise() {
        let a = Array3::<f64>::ones((2, 3, 4));
        let b = Array3::<f64>::from_elem((2, 3, 4), 3.0);
        let c = einsum_3d("ijk,ijk->ijk", &a, &b).expect("einsum_3d elem ok");
        assert_eq!(c.shape(), &[2, 3, 4]);
        assert_abs_diff_eq!(c[[1, 2, 3]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_einsum_3d_bad_rank() {
        let a = Array3::<f64>::ones((2, 3, 4));
        let b = Array3::<f64>::ones((2, 3, 4));
        // Output has wrong rank (2, not 3)
        assert!(einsum_3d("ijk,ijk->ij", &a, &b).is_err());
    }

    // --- tensor_mode_product ---

    #[test]
    fn test_mode_product_identity() {
        let t = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as f64);
        let eye = Array2::<f64>::eye(2);
        let out = tensor_mode_product(&t, &eye, 0).expect("mode_product ok");
        assert_eq!(out.shape(), &[2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_abs_diff_eq!(out[[i, j, k]], t[[i, j, k]], epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_mode_product_reduces_mode1() {
        // Sum along mode 1: ones(1,3) * tensor => shape (2,1,4)
        let t = Array3::<f64>::ones((2, 3, 4));
        let sumrow = Array2::from_elem((1, 3), 1.0_f64);
        let out = tensor_mode_product(&t, &sumrow, 1).expect("mode_product reduce ok");
        assert_eq!(out.shape(), &[2, 1, 4]);
        // Each element should be 3.0 (sum of three ones)
        for i in 0..2 {
            for k in 0..4 {
                assert_abs_diff_eq!(out[[i, 0, k]], 3.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_mode_product_shape_error() {
        let t = Array3::<f64>::ones((2, 3, 4));
        let bad_matrix = Array2::<f64>::eye(5); // 5×5, can't multiply on mode 0 (dim 2)
        assert!(tensor_mode_product(&t, &bad_matrix, 0).is_err());
    }

    // --- unfold_tensor / fold_tensor ---

    #[test]
    fn test_unfold_shapes() {
        let t = Array3::<f64>::zeros((2, 3, 4));
        assert_eq!(unfold_tensor(&t, 0).expect("mode 0").shape(), &[2, 12]);
        assert_eq!(unfold_tensor(&t, 1).expect("mode 1").shape(), &[3, 8]);
        assert_eq!(unfold_tensor(&t, 2).expect("mode 2").shape(), &[4, 6]);
    }

    #[test]
    fn test_unfold_bad_mode() {
        let t = Array3::<f64>::zeros((2, 3, 4));
        assert!(unfold_tensor(&t, 3).is_err());
    }

    #[test]
    fn test_fold_unfold_roundtrip() {
        let t = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as f64);
        for mode in 0..3 {
            let mat = unfold_tensor(&t, mode).expect("unfold ok");
            let t2 = fold_tensor(&mat, [2, 3, 4], mode).expect("fold ok");
            for i in 0..2 {
                for j in 0..3 {
                    for k in 0..4 {
                        assert_abs_diff_eq!(t[[i, j, k]], t2[[i, j, k]], epsilon = 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_fold_shape_mismatch() {
        let mat = Array2::<f64>::zeros((3, 8)); // Correct for mode-1, shape (2,3,4)
                                                // Wrong: try to fold into shape (2,4,4) on mode 1
        assert!(fold_tensor(&mat, [2, 4, 4], 1).is_err());
    }

    // --- khatri_rao ---

    #[test]
    fn test_khatri_rao_basic() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]]; // 2×2
        let b = array![[5.0_f64, 6.0], [7.0, 8.0]]; // 2×2
        let kr = khatri_rao(&a, &b).expect("khatri_rao ok");
        assert_eq!(kr.shape(), &[4, 2]);
        // Col 0: kron([1,3], [5,7]) = [5, 7, 15, 21]
        assert_abs_diff_eq!(kr[[0, 0]], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(kr[[1, 0]], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(kr[[2, 0]], 15.0, epsilon = 1e-10);
        assert_abs_diff_eq!(kr[[3, 0]], 21.0, epsilon = 1e-10);
        // Col 1: kron([2,4], [6,8]) = [12, 16, 24, 32]
        assert_abs_diff_eq!(kr[[0, 1]], 12.0, epsilon = 1e-10);
        assert_abs_diff_eq!(kr[[3, 1]], 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_khatri_rao_col_mismatch() {
        let a = array![[1.0_f64, 2.0]]; // 1×2
        let b = array![[1.0_f64, 2.0, 3.0]]; // 1×3
        assert!(khatri_rao(&a, &b).is_err());
    }

    // --- hadamard ---

    #[test]
    fn test_hadamard_basic() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[2.0_f64, 3.0], [4.0, 5.0]];
        let h = hadamard(&a, &b).expect("hadamard ok");
        assert_abs_diff_eq!(h[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(h[[0, 1]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(h[[1, 0]], 12.0, epsilon = 1e-10);
        assert_abs_diff_eq!(h[[1, 1]], 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hadamard_shape_mismatch() {
        let a = array![[1.0_f64, 2.0]]; // 1×2
        let b = array![[1.0_f64], [2.0]]; // 2×1
        assert!(hadamard(&a, &b).is_err());
    }

    // --- kronecker ---

    #[test]
    fn test_kronecker_basic() {
        // I2 ⊗ A = block diag(A, A)
        let eye = Array2::<f64>::eye(2);
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let k = kronecker(&eye, &a).expect("kronecker ok");
        assert_eq!(k.shape(), &[4, 4]);
        // Top-left block = a
        assert_abs_diff_eq!(k[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(k[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(k[[1, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(k[[1, 1]], 4.0, epsilon = 1e-10);
        // Off-diagonal blocks = zeros (from 0*a)
        assert_abs_diff_eq!(k[[0, 2]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(k[[0, 3]], 0.0, epsilon = 1e-10);
        // Bottom-right block = a
        assert_abs_diff_eq!(k[[2, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(k[[3, 3]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kronecker_mixed_sizes() {
        let a = array![[1.0_f64, 2.0]]; // 1×2
        let b = array![[3.0_f64], [4.0], [5.0]]; // 3×1
        let k = kronecker(&a, &b).expect("kronecker mixed ok");
        assert_eq!(k.shape(), &[3, 2]);
        assert_abs_diff_eq!(k[[0, 0]], 3.0, epsilon = 1e-10); // a[0,0] * b[0,0]
        assert_abs_diff_eq!(k[[1, 0]], 4.0, epsilon = 1e-10); // a[0,0] * b[1,0]
        assert_abs_diff_eq!(k[[2, 0]], 5.0, epsilon = 1e-10); // a[0,0] * b[2,0]
        assert_abs_diff_eq!(k[[0, 1]], 6.0, epsilon = 1e-10); // a[0,1] * b[0,0]
        assert_abs_diff_eq!(k[[1, 1]], 8.0, epsilon = 1e-10); // a[0,1] * b[1,0]
        assert_abs_diff_eq!(k[[2, 1]], 10.0, epsilon = 1e-10); // a[0,1] * b[2,0]
    }
}
