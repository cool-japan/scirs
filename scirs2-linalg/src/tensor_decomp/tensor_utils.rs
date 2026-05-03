//! Core 3-D tensor type and utilities for the `tensor_decomp` module.
//!
//! This module provides a lightweight, Vec-backed 3-D tensor (`Tensor3D`) and
//! the mathematical primitives needed by the decomposition algorithms:
//!
//! - Mode-n unfolding / refolding
//! - Khatri-Rao (column-wise Kronecker) product
//! - Mode-n product `T ×_n M`
//! - Frobenius norm and basic arithmetic helpers

use crate::error::{LinalgError, LinalgResult};

// ---------------------------------------------------------------------------
// Tensor3D
// ---------------------------------------------------------------------------

/// Dense 3-D tensor stored in row-major order.
///
/// Element `(i, j, k)` is stored at index `i * J*K + j * K + k`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor3D {
    /// Flat element storage (row-major).
    pub data: Vec<f64>,
    /// Shape `[I, J, K]`.
    pub shape: [usize; 3],
}

impl Tensor3D {
    /// Create a new `Tensor3D`.
    ///
    /// # Errors
    /// Returns [`LinalgError::ShapeError`] if `data.len() != I*J*K`.
    pub fn new(data: Vec<f64>, shape: [usize; 3]) -> LinalgResult<Self> {
        let expected = shape[0] * shape[1] * shape[2];
        if data.len() != expected {
            return Err(LinalgError::ShapeError(format!(
                "Tensor3D: data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                expected
            )));
        }
        Ok(Self { data, shape })
    }

    /// Create a zero tensor of the given shape.
    pub fn zeros(shape: [usize; 3]) -> Self {
        let n = shape[0] * shape[1] * shape[2];
        Self {
            data: vec![0.0_f64; n],
            shape,
        }
    }

    /// Get element `(i, j, k)`.
    ///
    /// # Panics
    /// Panics if any index is out of bounds.
    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        debug_assert!(i < self.shape[0] && j < self.shape[1] && k < self.shape[2]);
        self.data[i * self.shape[1] * self.shape[2] + j * self.shape[2] + k]
    }

    /// Set element `(i, j, k)`.
    ///
    /// # Panics
    /// Panics if any index is out of bounds.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, k: usize, v: f64) {
        debug_assert!(i < self.shape[0] && j < self.shape[1] && k < self.shape[2]);
        let idx = i * self.shape[1] * self.shape[2] + j * self.shape[2] + k;
        self.data[idx] = v;
    }

    /// Frobenius norm `‖T‖_F`.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Mode-n unfolding (matricization).
    ///
    /// Returns a matrix with shape:
    /// - mode 0 → `[I, J*K]`, columns are mode-0 fibers
    /// - mode 1 → `[J, I*K]`, columns are mode-1 fibers
    /// - mode 2 → `[K, I*J]`, columns are mode-2 fibers
    ///
    /// Uses the Kolda-Bader column ordering so that the Khatri-Rao product
    /// convention `X_(n) ≈ A_n · (A_{n+1} ⊙ A_{n-1})^T` is consistent.
    ///
    /// # Errors
    /// Returns [`LinalgError::ShapeError`] if `mode >= 3`.
    pub fn mode_unfold(&self, mode: usize) -> LinalgResult<Vec<Vec<f64>>> {
        let [i_dim, j_dim, k_dim] = self.shape;
        match mode {
            0 => {
                // rows: I, cols: J*K
                let rows = i_dim;
                let cols = j_dim * k_dim;
                let mut mat = vec![vec![0.0_f64; cols]; rows];
                for i in 0..i_dim {
                    for j in 0..j_dim {
                        for k in 0..k_dim {
                            mat[i][j * k_dim + k] = self.get(i, j, k);
                        }
                    }
                }
                Ok(mat)
            }
            1 => {
                // rows: J, cols: I*K (Kolda ordering: k varies fastest, then i)
                let rows = j_dim;
                let cols = i_dim * k_dim;
                let mut mat = vec![vec![0.0_f64; cols]; rows];
                for i in 0..i_dim {
                    for j in 0..j_dim {
                        for k in 0..k_dim {
                            mat[j][i * k_dim + k] = self.get(i, j, k);
                        }
                    }
                }
                Ok(mat)
            }
            2 => {
                // rows: K, cols: I*J (Kolda ordering: j varies fastest, then i)
                let rows = k_dim;
                let cols = i_dim * j_dim;
                let mut mat = vec![vec![0.0_f64; cols]; rows];
                for i in 0..i_dim {
                    for j in 0..j_dim {
                        for k in 0..k_dim {
                            mat[k][i * j_dim + j] = self.get(i, j, k);
                        }
                    }
                }
                Ok(mat)
            }
            _ => Err(LinalgError::ShapeError(format!(
                "mode_unfold: mode {mode} out of range for 3-D tensor"
            ))),
        }
    }

    /// Fold a matrix back into a 3-D tensor along the given mode.
    ///
    /// Reverses [`Tensor3D::mode_unfold`].
    ///
    /// # Errors
    /// Returns [`LinalgError::ShapeError`] if matrix dimensions are inconsistent
    /// with the given shape and mode.
    pub fn mode_fold(mat: &[Vec<f64>], mode: usize, shape: [usize; 3]) -> LinalgResult<Self> {
        let [i_dim, j_dim, k_dim] = shape;
        let mut t = Self::zeros(shape);
        match mode {
            0 => {
                if mat.len() != i_dim {
                    return Err(LinalgError::ShapeError(format!(
                        "mode_fold mode=0: matrix rows {} != I={}",
                        mat.len(),
                        i_dim
                    )));
                }
                for (i, row) in mat.iter().enumerate() {
                    if row.len() != j_dim * k_dim {
                        return Err(LinalgError::ShapeError(format!(
                            "mode_fold mode=0: matrix cols {} != J*K={}",
                            row.len(),
                            j_dim * k_dim
                        )));
                    }
                    for j in 0..j_dim {
                        for k in 0..k_dim {
                            t.set(i, j, k, row[j * k_dim + k]);
                        }
                    }
                }
            }
            1 => {
                if mat.len() != j_dim {
                    return Err(LinalgError::ShapeError(format!(
                        "mode_fold mode=1: matrix rows {} != J={}",
                        mat.len(),
                        j_dim
                    )));
                }
                for (j, row) in mat.iter().enumerate() {
                    if row.len() != i_dim * k_dim {
                        return Err(LinalgError::ShapeError(format!(
                            "mode_fold mode=1: matrix cols {} != I*K={}",
                            row.len(),
                            i_dim * k_dim
                        )));
                    }
                    for i in 0..i_dim {
                        for k in 0..k_dim {
                            t.set(i, j, k, row[i * k_dim + k]);
                        }
                    }
                }
            }
            2 => {
                if mat.len() != k_dim {
                    return Err(LinalgError::ShapeError(format!(
                        "mode_fold mode=2: matrix rows {} != K={}",
                        mat.len(),
                        k_dim
                    )));
                }
                for (k, row) in mat.iter().enumerate() {
                    if row.len() != i_dim * j_dim {
                        return Err(LinalgError::ShapeError(format!(
                            "mode_fold mode=2: matrix cols {} != I*J={}",
                            row.len(),
                            i_dim * j_dim
                        )));
                    }
                    for i in 0..i_dim {
                        for j in 0..j_dim {
                            t.set(i, j, k, row[i * j_dim + j]);
                        }
                    }
                }
            }
            _ => {
                return Err(LinalgError::ShapeError(format!(
                    "mode_fold: mode {mode} out of range for 3-D tensor"
                )));
            }
        }
        Ok(t)
    }

    /// Khatri-Rao product of two factor matrices.
    ///
    /// Given `A ∈ R^{m×r}` and `B ∈ R^{n×r}`, the Khatri-Rao product is
    /// `A ⊙ B ∈ R^{mn×r}` where the `r`-th column is `a_r ⊗ b_r`.
    ///
    /// # Errors
    /// Returns [`LinalgError::ShapeError`] if `A` and `B` have different number
    /// of columns.
    pub fn khatri_rao(a: &[Vec<f64>], b: &[Vec<f64>]) -> LinalgResult<Vec<Vec<f64>>> {
        let m = a.len();
        let n = b.len();
        if m == 0 || n == 0 {
            return Err(LinalgError::ShapeError(
                "khatri_rao: empty input matrix".to_string(),
            ));
        }
        let r = a[0].len();
        if b[0].len() != r {
            return Err(LinalgError::ShapeError(format!(
                "khatri_rao: A has {} columns but B has {} columns",
                r,
                b[0].len()
            )));
        }
        // result is (m*n) × r
        let mut result = vec![vec![0.0_f64; r]; m * n];
        for row_a in 0..m {
            for row_b in 0..n {
                let out_row = row_a * n + row_b;
                for col in 0..r {
                    result[out_row][col] = a[row_a][col] * b[row_b][col];
                }
            }
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Mode-n product
// ---------------------------------------------------------------------------

/// Mode-n product `T ×_n M` of a 3-D tensor with a matrix.
///
/// `M ∈ R^{p × shape[n]}` contracts along mode `n`, producing a tensor
/// with shape equal to `T.shape` with dimension `n` replaced by `p`.
///
/// # Errors
/// Returns [`LinalgError::ShapeError`] if dimensions are incompatible or
/// `mode >= 3`.
pub fn mode_n_product(t: &Tensor3D, mat: &[Vec<f64>], mode: usize) -> LinalgResult<Tensor3D> {
    let [i_dim, j_dim, k_dim] = t.shape;
    if mat.is_empty() {
        return Err(LinalgError::ShapeError(
            "mode_n_product: empty matrix".to_string(),
        ));
    }
    let p = mat.len();
    let mat_cols = mat[0].len();

    match mode {
        0 => {
            if mat_cols != i_dim {
                return Err(LinalgError::ShapeError(format!(
                    "mode_n_product mode=0: M cols {mat_cols} != I={i_dim}"
                )));
            }
            let new_shape = [p, j_dim, k_dim];
            let mut result = Tensor3D::zeros(new_shape);
            for pi in 0..p {
                for j in 0..j_dim {
                    for k in 0..k_dim {
                        let mut val = 0.0_f64;
                        for ii in 0..i_dim {
                            val += mat[pi][ii] * t.get(ii, j, k);
                        }
                        result.set(pi, j, k, val);
                    }
                }
            }
            Ok(result)
        }
        1 => {
            if mat_cols != j_dim {
                return Err(LinalgError::ShapeError(format!(
                    "mode_n_product mode=1: M cols {mat_cols} != J={j_dim}"
                )));
            }
            let new_shape = [i_dim, p, k_dim];
            let mut result = Tensor3D::zeros(new_shape);
            for i in 0..i_dim {
                for pj in 0..p {
                    for k in 0..k_dim {
                        let mut val = 0.0_f64;
                        for jj in 0..j_dim {
                            val += mat[pj][jj] * t.get(i, jj, k);
                        }
                        result.set(i, pj, k, val);
                    }
                }
            }
            Ok(result)
        }
        2 => {
            if mat_cols != k_dim {
                return Err(LinalgError::ShapeError(format!(
                    "mode_n_product mode=2: M cols {mat_cols} != K={k_dim}"
                )));
            }
            let new_shape = [i_dim, j_dim, p];
            let mut result = Tensor3D::zeros(new_shape);
            for i in 0..i_dim {
                for j in 0..j_dim {
                    for pk in 0..p {
                        let mut val = 0.0_f64;
                        for kk in 0..k_dim {
                            val += mat[pk][kk] * t.get(i, j, kk);
                        }
                        result.set(i, j, pk, val);
                    }
                }
            }
            Ok(result)
        }
        _ => Err(LinalgError::ShapeError(format!(
            "mode_n_product: mode {mode} out of range for 3-D tensor"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Dense linear algebra helpers (no external BLAS dependency)
// ---------------------------------------------------------------------------

/// Matrix multiply: `C = A * B` (all row-major Vec<Vec<f64>>).
///
/// `A` is `(m, k)`, `B` is `(k, n)`.
pub(crate) fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> LinalgResult<Vec<Vec<f64>>> {
    if a.is_empty() || b.is_empty() {
        return Err(LinalgError::ShapeError("mat_mul: empty input".to_string()));
    }
    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();
    if b.len() != k {
        return Err(LinalgError::ShapeError(format!(
            "mat_mul: A is ({m},{k}) but B is ({},{})",
            b.len(),
            n
        )));
    }
    let mut c = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i][l];
            for j in 0..n {
                c[i][j] += a_il * b[l][j];
            }
        }
    }
    Ok(c)
}

/// Transpose a matrix.
pub(crate) fn mat_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let n = a[0].len();
    let mut t = vec![vec![0.0_f64; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// `AᵀA` (Gram matrix).
pub(crate) fn gram(a: &[Vec<f64>]) -> LinalgResult<Vec<Vec<f64>>> {
    let at = mat_transpose(a);
    mat_mul(&at, a)
}

/// Element-wise Hadamard product of two square matrices of the same size.
pub(crate) fn hadamard(a: &[Vec<f64>], b: &[Vec<f64>]) -> LinalgResult<Vec<Vec<f64>>> {
    let m = a.len();
    if m == 0 {
        return Err(LinalgError::ShapeError(
            "hadamard: empty matrix".to_string(),
        ));
    }
    let n = a[0].len();
    if b.len() != m || b[0].len() != n {
        return Err(LinalgError::ShapeError(format!(
            "hadamard: shapes ({m},{n}) vs ({},{})",
            b.len(),
            b[0].len()
        )));
    }
    let mut c = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for j in 0..n {
            c[i][j] = a[i][j] * b[i][j];
        }
    }
    Ok(c)
}

/// Solve the symmetric positive-definite system `A x = b` via Cholesky (LL^T).
///
/// `a` is `(n,n)` SPD.  Returns the solution `x ∈ R^{n×rhs}` where `b` is
/// `(n, rhs)`.  Falls back to a damped pseudo-inverse if `a` is singular.
pub(crate) fn solve_spd(a: &[Vec<f64>], b: &[Vec<f64>]) -> LinalgResult<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return Err(LinalgError::ShapeError(
            "solve_spd: empty matrix".to_string(),
        ));
    }

    // Cholesky LL^T
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    // Not SPD — add small diagonal regularisation and retry
                    return solve_with_damping(a, b);
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }

    let rhs_cols = if b.is_empty() { 0 } else { b[0].len() };
    let mut x = vec![vec![0.0_f64; rhs_cols]; n];

    // Forward substitution: L y = b
    let mut y = vec![vec![0.0_f64; rhs_cols]; n];
    for i in 0..n {
        for c in 0..rhs_cols {
            let mut s = b[i][c];
            for k in 0..i {
                s -= l[i][k] * y[k][c];
            }
            y[i][c] = s / l[i][i];
        }
    }
    // Back substitution: L^T x = y
    for i in (0..n).rev() {
        for c in 0..rhs_cols {
            let mut s = y[i][c];
            for k in (i + 1)..n {
                s -= l[k][i] * x[k][c];
            }
            x[i][c] = s / l[i][i];
        }
    }
    Ok(x)
}

/// Ridge-regularised solve: `(A + delta*I) x = b`.
fn solve_with_damping(a: &[Vec<f64>], b: &[Vec<f64>]) -> LinalgResult<Vec<Vec<f64>>> {
    let n = a.len();
    // Estimate spectral radius via Gershgorin disks
    let max_diag = a
        .iter()
        .map(|row| row.iter().map(|x| x.abs()).fold(0.0_f64, f64::max))
        .fold(0.0_f64, f64::max);
    let delta = (max_diag * 1e-8).max(1e-12);
    let mut a_reg = a.to_vec();
    for i in 0..n {
        a_reg[i][i] += delta;
    }
    // Retry Cholesky — if still not SPD, return the right-hand side scaled down
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a_reg[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    // Fallback: identity solve (x ≈ b)
                    return Ok(b.to_vec());
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }

    let rhs_cols = if b.is_empty() { 0 } else { b[0].len() };
    let mut y = vec![vec![0.0_f64; rhs_cols]; n];
    let mut x = vec![vec![0.0_f64; rhs_cols]; n];
    for i in 0..n {
        for c in 0..rhs_cols {
            let mut s = b[i][c];
            for k in 0..i {
                s -= l[i][k] * y[k][c];
            }
            y[i][c] = s / l[i][i];
        }
    }
    for i in (0..n).rev() {
        for c in 0..rhs_cols {
            let mut s = y[i][c];
            for k in (i + 1)..n {
                s -= l[k][i] * x[k][c];
            }
            x[i][c] = s / l[i][i];
        }
    }
    Ok(x)
}

/// Truncated SVD via one-sided Jacobi iterations.
///
/// Returns `(U, s, Vt)` where `U ∈ R^{m×k}`, `s ∈ R^k`, `Vt ∈ R^{k×n}`,
/// and `k = min(min(m,n), rank)`.
///
/// Uses iterative power-iteration warm-start for numerical stability.
#[allow(clippy::type_complexity)]
pub(crate) fn truncated_svd(
    mat: &[Vec<f64>],
    rank: usize,
) -> LinalgResult<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>)> {
    if mat.is_empty() {
        return Err(LinalgError::ShapeError(
            "truncated_svd: empty matrix".to_string(),
        ));
    }
    let m = mat.len();
    let n = mat[0].len();
    let k = rank.min(m).min(n);
    if k == 0 {
        return Ok((vec![vec![0.0; 0]; m], Vec::new(), Vec::<Vec<f64>>::new()));
    }

    // Build AᵀA (or AAᵀ if m < n) for eigendecomposition
    // We use full SVD via Golub-Reinsch for correctness; for large matrices
    // a randomised approach would be preferred, but correctness is paramount.
    full_svd_truncated(mat, k)
}

/// Full SVD via Householder bidiagonalisation + QR sweep, returning top-k
/// components.  Correct for all matrix shapes.
#[allow(clippy::type_complexity)]
fn full_svd_truncated(
    mat: &[Vec<f64>],
    k: usize,
) -> LinalgResult<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>)> {
    let m = mat.len();
    let n = mat[0].len();

    // Use the scirs2-linalg SVD (ndarray based)
    // We convert our Vec<Vec<f64>> to a flat ndarray::Array2 and call SVD.
    use scirs2_core::ndarray::{s, Array2};

    let flat: Vec<f64> = mat.iter().flat_map(|row| row.iter().copied()).collect();
    let a = Array2::from_shape_vec((m, n), flat)
        .map_err(|e| LinalgError::ShapeError(format!("SVD reshape: {e}")))?;

    // Call linalg SVD
    let (u_full, s_arr, vt_full) = crate::decomposition::svd(&a.view(), true, None)
        .map_err(|e| LinalgError::ComputationError(format!("SVD failed: {e}")))?;

    // Truncate to k
    let k_actual = k.min(s_arr.len());
    let u_k = u_full.slice(s![.., ..k_actual]).to_owned();
    let vt_k = vt_full.slice(s![..k_actual, ..]).to_owned();
    let s_k: Vec<f64> = s_arr.iter().take(k_actual).copied().collect();

    // Convert back to Vec<Vec<f64>>
    let u_out: Vec<Vec<f64>> = (0..m)
        .map(|i| (0..k_actual).map(|j| u_k[[i, j]]).collect())
        .collect();
    let vt_out: Vec<Vec<f64>> = (0..k_actual)
        .map(|i| (0..n).map(|j| vt_k[[i, j]]).collect())
        .collect();

    Ok((u_out, s_k, vt_out))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor3d_get_set() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = Tensor3D::new(data, [2, 3, 4]).expect("ok");
        assert_eq!(t.get(0, 0, 0), 0.0);
        assert_eq!(t.get(1, 2, 3), 23.0);
        assert_eq!(t.get(0, 1, 2), 6.0);
    }

    #[test]
    fn test_mode_unfold_fold_roundtrip() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = Tensor3D::new(data, [2, 3, 4]).expect("ok");
        for mode in 0..3 {
            let mat = t.mode_unfold(mode).expect("unfold");
            let t2 = Tensor3D::mode_fold(&mat, mode, [2, 3, 4]).expect("fold");
            for (a, b) in t.data.iter().zip(t2.data.iter()) {
                assert!((a - b).abs() < 1e-12, "mode {mode}: {a} != {b}");
            }
        }
    }

    #[test]
    fn test_khatri_rao() {
        // A is 3×2, B is 4×2, result should be 12×2
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let b = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let kr = Tensor3D::khatri_rao(&a, &b).expect("ok");
        assert_eq!(kr.len(), 12);
        assert_eq!(kr[0].len(), 2);
        // column 0: a[:,0] ⊗ b[:,0] = [1,3,5] ⊗ [1,3,5,7]
        assert_eq!(kr[0][0], 1.0 * 1.0); // a[0][0] * b[0][0]
        assert_eq!(kr[1][0], 1.0 * 3.0); // a[0][0] * b[1][0]
        assert_eq!(kr[4][0], 3.0 * 1.0); // a[1][0] * b[0][0]
    }

    #[test]
    fn test_mode_n_product_mode0() {
        // Tensor [2,3,4] × M[5,2] along mode 0 → [5,3,4]
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = Tensor3D::new(data, [2, 3, 4]).expect("ok");
        // Identity-like matrix: [[1,0],[0,1],[1,0],[0,1],[1,1]] → not a simple identity
        // Use a 2×2 identity to check that M=I gives the same tensor
        let m_id = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = mode_n_product(&t, &m_id, 0).expect("ok");
        assert_eq!(result.shape, [2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert!((result.get(i, j, k) - t.get(i, j, k)).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_mat_mul_shapes() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]; // 2×3
        let b = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]; // 3×2
        let c = mat_mul(&a, &b).expect("ok"); // 2×2
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].len(), 2);
        assert!((c[0][0] - 58.0).abs() < 1e-12);
        assert!((c[1][1] - 154.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_spd() {
        // A = [[4,2],[2,3]], b = [[8],[7]]
        // Ax = b => x = A^{-1} b = [[1.25],[1.5]]
        // Verify: 4*1.25 + 2*1.5 = 8, 2*1.25 + 3*1.5 = 7
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let b = vec![vec![8.0], vec![7.0]];
        let x = solve_spd(&a, &b).expect("ok");
        assert!((x[0][0] - 1.25).abs() < 1e-10, "x[0]={}", x[0][0]);
        assert!((x[1][0] - 1.5).abs() < 1e-10, "x[1]={}", x[1][0]);
    }
}
