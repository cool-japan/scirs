//! Core tensor type: N-way dense array with explicit strides.
//!
//! [`Tensor<F>`] is the foundational data structure for the `tensor` module.
//! It stores elements in a flat `Vec<F>` and tracks the shape/strides, enabling
//! efficient mode-n matricization, Tucker n-mode products, and norm computations
//! that are required by the higher-level decomposition algorithms.
//!
//! All methods return `Result` rather than panicking.  The only arithmetic that
//! can produce a non-finite result is division-by-zero in
//! [`Tensor::frobenius_norm`], which is guarded and returns `F::zero()` for
//! an empty tensor.
//!
//! ## Row-major (C) layout
//!
//! The default constructor ([`Tensor::new`] and [`Tensor::zeros`]) builds a
//! *contiguous* row-major tensor where:
//!
//! ```text
//! strides[d] = product(shape[d+1 ..])
//! ```
//!
//! The strides field is kept public so that future slicing or view operations
//! can be implemented without changing the API.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Float bound alias
// ---------------------------------------------------------------------------

/// Trait alias for all scalar types accepted by the `tensor` module.
pub trait TensorScalar: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}
impl<T> TensorScalar for T where T: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}

// ---------------------------------------------------------------------------
// Tensor struct
// ---------------------------------------------------------------------------

/// Dense N-way tensor with explicit strides.
///
/// # Layout
///
/// Elements are stored in row-major (C-contiguous) order for freshly
/// constructed tensors.  The stride for dimension `d` is the number of
/// elements you skip in `data` to advance one step in dimension `d`.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::tensor::core::Tensor;
///
/// // 2 × 3 matrix stored as a rank-2 tensor
/// let t: Tensor<f64> = Tensor::new(
///     (0..6).map(|x| x as f64).collect(),
///     vec![2, 3],
/// ).expect("valid shape");
/// assert_eq!(t.get(&[1, 2]).expect("in bounds"), 5.0);
/// ```
#[derive(Debug, Clone)]
pub struct Tensor<F> {
    /// Flat storage in row-major order.
    pub data: Vec<F>,
    /// Shape of each dimension.
    pub shape: Vec<usize>,
    /// Strides (number of `data` elements per step in each dimension).
    pub strides: Vec<usize>,
}

impl<F: TensorScalar> Tensor<F> {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Create a tensor from a flat data buffer and a shape.
    ///
    /// Strides are computed as row-major (C) strides:
    /// `strides[d] = shape[d+1] * shape[d+2] * ... * shape[N-1]`.
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::ShapeError`] when `data.len()` does not equal
    /// the product of `shape`.
    pub fn new(data: Vec<F>, shape: Vec<usize>) -> LinalgResult<Self> {
        let total: usize = if shape.is_empty() {
            0
        } else {
            shape.iter().product()
        };
        if data.len() != total {
            return Err(LinalgError::ShapeError(format!(
                "Data length {} does not match shape {:?} (product {})",
                data.len(),
                shape,
                total
            )));
        }
        let strides = compute_row_major_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Create an all-zero tensor with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total: usize = if shape.is_empty() {
            0
        } else {
            shape.iter().product()
        };
        let strides = compute_row_major_strides(&shape);
        Self {
            data: vec![F::zero(); total],
            shape,
            strides,
        }
    }

    // ------------------------------------------------------------------
    // Metadata
    // ------------------------------------------------------------------

    /// Number of dimensions (order / rank of the tensor).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    // ------------------------------------------------------------------
    // Indexing
    // ------------------------------------------------------------------

    /// Read an element at a multi-dimensional index.
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::IndexError`] when `indices` has the wrong length
    /// or any index is out of bounds.
    pub fn get(&self, indices: &[usize]) -> LinalgResult<F> {
        let flat = self.flat_index(indices)?;
        Ok(self.data[flat])
    }

    /// Write an element at a multi-dimensional index.
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::IndexError`] when `indices` has the wrong length
    /// or any index is out of bounds.
    pub fn set(&mut self, indices: &[usize], val: F) -> LinalgResult<()> {
        let flat = self.flat_index(indices)?;
        self.data[flat] = val;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Matricization (unfolding)
    // ------------------------------------------------------------------

    /// Mode-n matricization (unfolding) of the tensor.
    ///
    /// The mode-`n` unfolding maps the tensor to a 2-D matrix where:
    ///
    /// - **Rows** → dimension `n`  (size `shape[n]`)
    /// - **Columns** → all remaining dimensions combined (row-major, modes
    ///   `[0, 1, …, n-1, n+1, …, N-1]`)
    ///
    /// # Returns
    ///
    /// `Array2<F>` of shape `(shape[n], numel / shape[n])`.
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::IndexError`] when `mode >= ndim()`.
    pub fn unfold(&self, mode: usize) -> LinalgResult<Array2<F>> {
        let ndim = self.ndim();
        if mode >= ndim {
            return Err(LinalgError::IndexError(format!(
                "Unfold mode {mode} out of range for {ndim}-D tensor"
            )));
        }
        let n_rows = self.shape[mode];
        let n_cols = self.numel() / n_rows;
        let mut mat = Array2::<F>::zeros((n_rows, n_cols));

        let numel = self.numel();
        for flat in 0..numel {
            let multi = self.multi_index(flat);
            let row = multi[mode];
            // column index: modes other than `mode` in natural order
            let mut col = 0usize;
            let mut stride = 1usize;
            for d in (0..ndim).rev() {
                if d == mode {
                    continue;
                }
                col += multi[d] * stride;
                stride *= self.shape[d];
            }
            mat[[row, col]] = self.data[flat];
        }
        Ok(mat)
    }

    /// Reconstruct a tensor from a mode-n unfolded matrix and a target shape.
    ///
    /// This is the inverse of [`unfold`].
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::ShapeError`] when `matrix` dimensions are
    /// incompatible with `shape` and `mode`.
    pub fn fold(matrix: &Array2<F>, mode: usize, shape: Vec<usize>) -> LinalgResult<Self> {
        let ndim = shape.len();
        if mode >= ndim {
            return Err(LinalgError::IndexError(format!(
                "Fold mode {mode} out of range for {ndim}-D shape"
            )));
        }
        let expected_rows = shape[mode];
        let total: usize = shape.iter().product();
        let expected_cols = total / expected_rows;
        if matrix.nrows() != expected_rows || matrix.ncols() != expected_cols {
            return Err(LinalgError::ShapeError(format!(
                "Matrix shape {:?} incompatible with fold mode {} shape {:?}",
                matrix.shape(),
                mode,
                shape
            )));
        }
        let strides = compute_row_major_strides(&shape);
        let mut data = vec![F::zero(); total];
        // Reverse the unfold column mapping
        for row in 0..expected_rows {
            for col in 0..expected_cols {
                // Recover multi-index from (row, col)
                let mut multi = vec![0usize; ndim];
                multi[mode] = row;
                let mut remaining = col;
                // modes in reverse order as in unfold
                let other_modes: Vec<usize> = (0..ndim).filter(|&d| d != mode).collect();
                // stride for col encoding is accumulated right-to-left
                let mut col_strides = vec![1usize; other_modes.len()];
                for k in (0..other_modes.len().saturating_sub(1)).rev() {
                    col_strides[k] = col_strides[k + 1] * shape[other_modes[k + 1]];
                }
                for k in 0..other_modes.len() {
                    multi[other_modes[k]] = remaining / col_strides[k];
                    remaining %= col_strides[k];
                }
                // Compute flat index
                let flat: usize = multi.iter().zip(strides.iter()).map(|(i, s)| i * s).sum();
                data[flat] = matrix[[row, col]];
            }
        }
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    // ------------------------------------------------------------------
    // Tucker n-mode product
    // ------------------------------------------------------------------

    /// Mode-n product of the tensor with a matrix.
    ///
    /// Computes `T ×_n M` where `M` has shape `(J, I_n)`:
    ///
    /// ```text
    /// (T ×_n M)(i_1, …, i_{n-1}, j, i_{n+1}, …) = Σ_{i_n} T(…, i_n, …) · M[j, i_n]
    /// ```
    ///
    /// The resulting tensor has the same shape as `self`, but dimension `mode`
    /// changes from `I_n` to `J`.
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::DimensionError`] when `M.ncols() != shape[mode]`.
    pub fn mode_product(&self, matrix: &Array2<F>, mode: usize) -> LinalgResult<Self> {
        let ndim = self.ndim();
        if mode >= ndim {
            return Err(LinalgError::IndexError(format!(
                "Mode {mode} out of range for {ndim}-D tensor"
            )));
        }
        if matrix.ncols() != self.shape[mode] {
            return Err(LinalgError::DimensionError(format!(
                "Matrix ncols {} != tensor shape[{mode}] = {}",
                matrix.ncols(),
                self.shape[mode]
            )));
        }
        let j = matrix.nrows();
        let mut new_shape = self.shape.clone();
        new_shape[mode] = j;
        let mut result = Self::zeros(new_shape);
        let numel = self.numel();

        for flat in 0..numel {
            let multi = self.multi_index(flat);
            let in_idx = multi[mode];
            let val = self.data[flat];
            for out_idx in 0..j {
                let m_val = matrix[[out_idx, in_idx]];
                let mut out_multi = multi.clone();
                out_multi[mode] = out_idx;
                let out_flat = flat_from_multi(&out_multi, &result.strides);
                result.data[out_flat] += val * m_val;
            }
        }
        Ok(result)
    }

    // ------------------------------------------------------------------
    // Norms
    // ------------------------------------------------------------------

    /// Frobenius norm: `sqrt(sum of squared elements)`.
    ///
    /// Returns `F::zero()` for an empty tensor.
    pub fn frobenius_norm(&self) -> F {
        let sq: F = self
            .data
            .iter()
            .map(|&x| x * x)
            .fold(F::zero(), |a, b| a + b);
        sq.sqrt()
    }

    /// Inner product of two tensors (element-wise dot product).
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::ShapeError`] when shapes differ.
    pub fn inner_product(&self, other: &Self) -> LinalgResult<F> {
        if self.shape != other.shape {
            return Err(LinalgError::ShapeError(format!(
                "Shape mismatch for inner product: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }
        let ip: F = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .fold(F::zero(), |acc, x| acc + x);
        Ok(ip)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Convert a flat index to a multi-dimensional index (row-major).
    pub(crate) fn multi_index(&self, mut flat: usize) -> Vec<usize> {
        let ndim = self.ndim();
        let mut idx = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            idx[d] = flat % self.shape[d];
            flat /= self.shape[d];
        }
        idx
    }

    /// Convert a multi-dimensional index to a flat index using `self.strides`.
    pub(crate) fn flat_index(&self, indices: &[usize]) -> LinalgResult<usize> {
        if indices.len() != self.shape.len() {
            return Err(LinalgError::IndexError(format!(
                "Index rank {} != tensor rank {}",
                indices.len(),
                self.shape.len()
            )));
        }
        let mut flat = 0usize;
        for (d, (&idx, &stride)) in indices.iter().zip(self.strides.iter()).enumerate() {
            if idx >= self.shape[d] {
                return Err(LinalgError::IndexError(format!(
                    "Index {} out of bounds for dimension {} (size {})",
                    idx, d, self.shape[d]
                )));
            }
            flat += idx * stride;
        }
        Ok(flat)
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Compute row-major strides for a given shape.
///
/// `strides[d] = product(shape[d+1 ..])`.
pub(crate) fn compute_row_major_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Compute a flat index from a multi-index and strides.
#[inline]
pub(crate) fn flat_from_multi(multi: &[usize], strides: &[usize]) -> usize {
    multi.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_234() -> Tensor<f64> {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        Tensor::new(data, vec![2, 3, 4]).expect("valid")
    }

    #[test]
    fn test_new_shape_mismatch() {
        let r = Tensor::<f64>::new(vec![1.0, 2.0], vec![3]);
        assert!(r.is_err());
    }

    #[test]
    fn test_get_set_roundtrip() {
        let mut t = make_234();
        t.set(&[1, 2, 3], 99.0).expect("in bounds");
        assert_abs_diff_eq!(t.get(&[1, 2, 3]).expect("ok"), 99.0, epsilon = 1e-12);
    }

    #[test]
    fn test_frobenius_norm() {
        // [1, 2, 3, 4] → sqrt(30)
        let t = Tensor::new(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4]).expect("ok");
        assert_abs_diff_eq!(t.frobenius_norm(), 30.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_inner_product() {
        let a = Tensor::new(vec![1.0_f64, 2.0, 3.0], vec![3]).expect("ok");
        let b = Tensor::new(vec![4.0_f64, 5.0, 6.0], vec![3]).expect("ok");
        assert_abs_diff_eq!(a.inner_product(&b).expect("ok"), 32.0, epsilon = 1e-12);
    }

    #[test]
    fn test_unfold_mode0() {
        let t = make_234(); // shape [2, 3, 4]
        let m = t.unfold(0).expect("ok");
        assert_eq!(m.shape(), &[2, 12]);
    }

    #[test]
    fn test_unfold_mode1() {
        let t = make_234();
        let m = t.unfold(1).expect("ok");
        assert_eq!(m.shape(), &[3, 8]);
    }

    #[test]
    fn test_fold_roundtrip() {
        let t = make_234();
        let m = t.unfold(1).expect("ok");
        let t2 = Tensor::fold(&m, 1, vec![2, 3, 4]).expect("ok");
        for i in 0..t.numel() {
            assert_abs_diff_eq!(t.data[i], t2.data[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_mode_product_shape() {
        let t = make_234(); // [2, 3, 4]
                            // mode-1 product with 5×3 matrix → shape [2, 5, 4]
        let mat = Array2::<f64>::zeros((5, 3));
        let result = t.mode_product(&mat, 1).expect("ok");
        assert_eq!(result.shape, vec![2, 5, 4]);
    }

    #[test]
    fn test_mode_product_identity() {
        let t = make_234(); // [2, 3, 4]
        let eye = Array2::<f64>::eye(3);
        let result = t.mode_product(&eye, 1).expect("identity");
        for i in 0..t.numel() {
            assert_abs_diff_eq!(t.data[i], result.data[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_zeros() {
        let z = Tensor::<f32>::zeros(vec![3, 4, 5]);
        assert_eq!(z.numel(), 60);
        assert!(z.data.iter().all(|&x| x == 0.0));
    }
}
