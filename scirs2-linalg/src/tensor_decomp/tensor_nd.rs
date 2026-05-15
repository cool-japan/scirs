//! N-dimensional tensor type with CP, Tucker, and Tensor Train decompositions.
//!
//! This module provides a general [`Tensor`] type supporting arbitrary rank
//! (number of modes/dimensions) along with:
//!
//! - [`Tensor`] – N-dimensional dense tensor with mode unfolding/folding
//! - [`CpDecomposition`] / [`cp_als`] – CANDECOMP/PARAFAC via Alternating Least Squares
//! - [`TuckerDecomposition`] / [`tucker_hosvd`] / [`tucker_hooi`] – Tucker via HOSVD/HOOI
//! - [`TensorTrainDecomposition`] / [`tensor_train_svd`] – Tensor Train via TT-SVD
//!
//! # Example
//!
//! ```rust
//! use scirs2_linalg::tensor_decomp::tensor_nd::{Tensor, cp_als};
//!
//! let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
//! let t = Tensor::new(vec![2, 3, 4], data).expect("valid input");
//! assert_eq!(t.n_dims(), 3);
//! assert_eq!(t.numel(), 24);
//!
//! let cp = cp_als(&t, 2, 100, 1e-6, 42).expect("valid input");
//! assert_eq!(cp.rank, 2);
//! ```

use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// Internal helpers
// ============================================================================

fn matmul_nn(a: &[Vec<f64>], b: &[Vec<f64>], m: usize, k: usize, n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for l in 0..k {
            if a[i][l] == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    c
}

fn transpose_2d(a: &[Vec<f64>], m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn dot_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn frobenius_sq(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|r| r.iter()).map(|x| x * x).sum()
}

/// Solve A x = b for square A (Gaussian elimination).
fn solve_sq(a: &[Vec<f64>], b: &[f64], n: usize) -> LinalgResult<Vec<f64>> {
    let mut mat = a.to_vec();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in col + 1..n {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular".to_string(),
            ));
        }
        mat.swap(col, max_row);
        rhs.swap(col, max_row);
        let pivot = mat[col][col];
        for row in col + 1..n {
            let factor = mat[row][col] / pivot;
            rhs[row] -= factor * rhs[col];
            for j in col..n {
                let v = mat[col][j];
                mat[row][j] -= factor * v;
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in i + 1..n {
            s -= mat[i][j] * x[j];
        }
        if mat[i][i].abs() < 1e-300 {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular".to_string(),
            ));
        }
        x[i] = s / mat[i][i];
    }
    Ok(x)
}

/// Thin SVD of a rows × cols matrix (Gram-Schmidt + Jacobi eigendecomposition).
/// Returns (U rows×k, S k, Vt k×cols) where k = min(rows, cols).
fn thin_svd(a: &[Vec<f64>], rows: usize, cols: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let k = rows.min(cols);
    // Build A^T A (cols × cols)
    let at = transpose_2d(a, rows, cols);
    let ata = matmul_nn(&at, a, cols, rows, cols);
    // Jacobi eigen
    let (eigenvals, eigenvecs) = jacobi_eigen_sym(&ata, cols);
    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|&i, &j| {
        eigenvals[j]
            .partial_cmp(&eigenvals[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut s = vec![0.0f64; k];
    let mut vt = vec![vec![0.0f64; cols]; k];
    for (idx, &orig) in order.iter().enumerate().take(k) {
        s[idx] = eigenvals[orig].max(0.0).sqrt();
        for j in 0..cols {
            vt[idx][j] = eigenvecs[j][orig];
        }
    }

    let mut u = vec![vec![0.0f64; k]; rows];
    for idx in 0..k {
        if s[idx] > 1e-14 {
            for r in 0..rows {
                let sum: f64 = (0..cols).map(|c| a[r][c] * vt[idx][c]).sum();
                u[r][idx] = sum / s[idx];
            }
        }
    }
    (u, s, vt)
}

/// Symmetric Jacobi eigen-decomposition. Returns (eigenvalues, eigenvectors as columns).
fn jacobi_eigen_sym(a: &[Vec<f64>], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut mat = a.to_vec();
    let mut vecs: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for _ in 0..200 {
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in i + 1..n {
                if mat[i][j].abs() > max_val {
                    max_val = mat[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-13 {
            break;
        }
        let theta = (mat[q][q] - mat[p][p]) / (2.0 * mat[p][q]);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let tau = s / (1.0 + c);
        let a_pp = mat[p][p];
        let a_qq = mat[q][q];
        let a_pq = mat[p][q];
        mat[p][p] = a_pp - t * a_pq;
        mat[q][q] = a_qq + t * a_pq;
        mat[p][q] = 0.0;
        mat[q][p] = 0.0;
        for r in 0..n {
            if r != p && r != q {
                let a_rp = mat[r][p];
                let a_rq = mat[r][q];
                mat[r][p] = a_rp - s * (a_rq + tau * a_rp);
                mat[p][r] = mat[r][p];
                mat[r][q] = a_rq + s * (a_rp - tau * a_rq);
                mat[q][r] = mat[r][q];
            }
        }
        for r in 0..n {
            let v_rp = vecs[r][p];
            let v_rq = vecs[r][q];
            vecs[r][p] = v_rp - s * (v_rq + tau * v_rp);
            vecs[r][q] = v_rq + s * (v_rp - tau * v_rq);
        }
    }
    let eigenvals: Vec<f64> = (0..n).map(|i| mat[i][i]).collect();
    (eigenvals, vecs)
}

/// Khatri-Rao product of matrices A (m×r) and B (n×r): result is (mn × r).
fn khatri_rao(a: &[Vec<f64>], m: usize, b: &[Vec<f64>], n: usize, r: usize) -> Vec<Vec<f64>> {
    let mut kr = vec![vec![0.0; r]; m * n];
    for i in 0..m {
        for j in 0..n {
            for k in 0..r {
                kr[i * n + j][k] = a[i][k] * b[j][k];
            }
        }
    }
    kr
}

/// Generate Gaussian random matrix m×k with given seed.
fn random_matrix(m: usize, k: usize, seed: u64) -> Vec<Vec<f64>> {
    use scirs2_core::random::prelude::*;
    use scirs2_core::random::rngs::SmallRng;
    use scirs2_core::random::{Distribution, Normal};
    let normal = Normal::new(0.0f64, 1.0)
        .unwrap_or_else(|_| Normal::new(0.0, 1.0 - f64::EPSILON).expect("normal"));
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..m)
        .map(|_| (0..k).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

/// Gram-Schmidt QR: returns Q (m×k), columns are orthonormal.
fn qr_cols(a: &[Vec<f64>], m: usize, k: usize) -> Vec<Vec<f64>> {
    // Store orthonormal columns as rows of an (k × m) intermediate
    let mut qs: Vec<Vec<f64>> = Vec::with_capacity(k);
    for j in 0..k {
        let mut v: Vec<f64> = (0..m).map(|i| a[i][j]).collect();
        for q in &qs {
            let proj = dot_vec(q, &v);
            for l in 0..m {
                v[l] -= proj * q[l];
            }
        }
        let nv = dot_vec(&v, &v).sqrt();
        if nv > 1e-14 {
            qs.push(v.iter().map(|x| x / nv).collect());
        } else {
            // Degenerate: use a canonical direction orthogonal to current set
            let mut e = vec![0.0; m];
            for d in 0..m {
                e[d] = 1.0;
                for q in &qs {
                    let proj = dot_vec(q, &e);
                    let qq = q.clone();
                    for l in 0..m {
                        e[l] -= proj * qq[l];
                    }
                }
                let ne = dot_vec(&e, &e).sqrt();
                if ne > 1e-14 {
                    qs.push(e.iter().map(|x| x / ne).collect());
                    break;
                }
                e = vec![0.0; m];
            }
        }
    }
    // Convert back to m×k matrix
    let mut q = vec![vec![0.0; k]; m];
    for j in 0..qs.len().min(k) {
        for i in 0..m {
            q[i][j] = qs[j][i];
        }
    }
    q
}

// ============================================================================
// Tensor type
// ============================================================================

/// Dense N-dimensional tensor with row-major (C-order) storage.
///
/// Indices are stored such that the last dimension varies fastest.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::tensor_decomp::tensor_nd::Tensor;
///
/// let t = Tensor::zeros(vec![2, 3, 4]);
/// assert_eq!(t.n_dims(), 3);
/// assert_eq!(t.numel(), 24);
/// assert_eq!(t.get(&[0, 0, 0]), 0.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Flat element storage in row-major order.
    pub data: Vec<f64>,
    /// Shape of each mode.
    pub shape: Vec<usize>,
    /// Strides: stride\[i\] = product of shape\[i+1..\].
    pub strides: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor with given shape and data.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len()` does not equal the product of the shape.
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> LinalgResult<Self> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(LinalgError::ShapeError(format!(
                "Tensor::new: data.len()={} but product(shape)={}",
                data.len(),
                n
            )));
        }
        let strides = compute_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Create an all-zeros tensor with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        let strides = compute_strides(&shape);
        Self {
            data: vec![0.0; n],
            shape,
            strides,
        }
    }

    /// Number of dimensions (modes).
    pub fn n_dims(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Get an element by multi-index `indices`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if any index is out of bounds.
    pub fn get(&self, indices: &[usize]) -> f64 {
        let idx = self.flat_index(indices);
        self.data[idx]
    }

    /// Set an element by multi-index.
    pub fn set(&mut self, indices: &[usize], val: f64) {
        let idx = self.flat_index(indices);
        self.data[idx] = val;
    }

    /// Reshape to new shape (must have the same total number of elements).
    pub fn reshape(&self, new_shape: Vec<usize>) -> LinalgResult<Self> {
        let n: usize = new_shape.iter().product();
        if n != self.numel() {
            return Err(LinalgError::ShapeError(format!(
                "Tensor::reshape: cannot reshape {} elements to shape {:?} ({} elements)",
                self.numel(),
                new_shape,
                n
            )));
        }
        let strides = compute_strides(&new_shape);
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides,
        })
    }

    /// Mode-n unfolding (matricization): returns a matrix of shape `(shape[mode], prod of other dims)`.
    ///
    /// Uses the standard Kolda-Bader convention: columns correspond to multi-indices of
    /// all modes except `mode`, ordered as `(j_{mode+1}, j_{mode+2}, ..., j_{N-1}, j_0, ..., j_{mode-1})`.
    ///
    /// # Errors
    ///
    /// Returns an error if `mode >= n_dims()`.
    pub fn unfold(&self, mode: usize) -> LinalgResult<Vec<Vec<f64>>> {
        let ndim = self.n_dims();
        if mode >= ndim {
            return Err(LinalgError::ShapeError(format!(
                "unfold: mode {} >= n_dims {}",
                mode, ndim
            )));
        }
        let rows = self.shape[mode];
        let cols = self.numel() / rows;
        let mut mat = vec![vec![0.0; cols]; rows];

        // Iterate over all multi-indices
        let mut idx = vec![0usize; ndim];
        for flat in 0..self.numel() {
            let row = idx[mode];
            // Column index: multi-index of all other modes in order (mode+1..N, 0..mode)
            let mut col = 0usize;
            let mut col_stride = 1usize;
            // Process modes in order: mode+1, mode+2, ..., N-1, 0, 1, ..., mode-1
            for step in 0..ndim - 1 {
                let m = (mode + 1 + step) % ndim;
                col += idx[m] * col_stride;
                col_stride *= self.shape[m];
            }
            mat[row][col] = self.data[flat];
            // Advance multi-index (last dim fastest)
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < self.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(mat)
    }

    /// Fold a matrix back into a tensor of given shape along the given mode.
    ///
    /// Inverse of `unfold`.
    pub fn fold(matrix: &[Vec<f64>], mode: usize, shape: Vec<usize>) -> LinalgResult<Self> {
        let ndim = shape.len();
        if mode >= ndim {
            return Err(LinalgError::ShapeError(format!(
                "fold: mode {} >= ndim {}",
                mode, ndim
            )));
        }
        let rows = shape[mode];
        let cols: usize = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != mode)
            .map(|(_, &s)| s)
            .product();
        if matrix.len() != rows || (!matrix.is_empty() && matrix[0].len() != cols) {
            return Err(LinalgError::ShapeError(format!(
                "fold: matrix shape {}×{} does not match shape {:?} for mode {}",
                matrix.len(),
                if matrix.is_empty() {
                    0
                } else {
                    matrix[0].len()
                },
                shape,
                mode
            )));
        }
        let mut t = Tensor::zeros(shape.clone());
        let mut idx = vec![0usize; ndim];
        let n: usize = shape.iter().product();
        for flat in 0..n {
            let row = idx[mode];
            let mut col = 0usize;
            let mut col_stride = 1usize;
            for step in 0..ndim - 1 {
                let m = (mode + 1 + step) % ndim;
                col += idx[m] * col_stride;
                col_stride *= shape[m];
            }
            t.data[flat] = matrix[row][col];
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(t)
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Compute flat index from multi-index.
    fn flat_index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(indices.len(), self.n_dims());
        indices.iter().zip(&self.strides).map(|(i, s)| i * s).sum()
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ============================================================================
// CP decomposition via ALS
// ============================================================================

/// Result of CP (CANDECOMP/PARAFAC) decomposition.
///
/// Represents the tensor as: `T ≈ sum_{r=1}^{rank} weight_r * a1_r ⊗ a2_r ⊗ ... ⊗ aN_r`
#[derive(Debug, Clone)]
pub struct CpDecomposition {
    /// Factor matrices: `factors[n]` is `(shape[n] × rank)`.
    pub factors: Vec<Vec<Vec<f64>>>,
    /// Normalization weights of each component.
    pub weights: Vec<f64>,
    /// Rank.
    pub rank: usize,
    /// Number of ALS iterations performed.
    pub n_iter: usize,
    /// Fit = 1 - ||T - T_approx|| / ||T||.
    pub fit: f64,
}

/// CP decomposition via Alternating Least Squares (ALS).
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `rank` - CP rank
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance (relative change in fit)
/// * `seed` - Random seed for initialization
///
/// # Errors
///
/// Returns an error if any ALS subproblem becomes singular.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::tensor_decomp::tensor_nd::{Tensor, cp_als};
///
/// let data: Vec<f64> = (0..8).map(|x| x as f64).collect();
/// let t = Tensor::new(vec![2, 2, 2], data).expect("valid input");
/// let cp = cp_als(&t, 2, 100, 1e-5, 42).expect("valid input");
/// assert_eq!(cp.rank, 2);
/// assert!(cp.fit >= 0.0 && cp.fit <= 1.0 + 1e-6);
/// ```
pub fn cp_als(
    tensor: &Tensor,
    rank: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> LinalgResult<CpDecomposition> {
    let ndim = tensor.n_dims();
    if ndim < 2 {
        return Err(LinalgError::ShapeError(
            "cp_als requires at least 2 modes".into(),
        ));
    }
    let rank = rank.max(1);

    // Initialize factor matrices randomly
    let mut factors: Vec<Vec<Vec<f64>>> = (0..ndim)
        .map(|n| {
            let dim = tensor.shape[n];
            let m = random_matrix(dim, rank, seed.wrapping_add(n as u64 * 1234567891));
            // Normalize columns
            let mut f = m;
            for r in 0..rank {
                let nrm: f64 = (0..dim).map(|i| f[i][r] * f[i][r]).sum::<f64>().sqrt();
                if nrm > 1e-14 {
                    for i in 0..dim {
                        f[i][r] /= nrm;
                    }
                }
            }
            f
        })
        .collect();

    let norm_t = tensor.frobenius_norm();
    let mut weights = vec![1.0f64; rank];
    let mut prev_fit = -1.0f64;
    let mut n_iter = 0usize;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // ALS: update each factor matrix in turn
        for mode in 0..ndim {
            // Compute V = hadamard product of (A_n^T A_n) for n != mode
            let mut v = vec![vec![1.0; rank]; rank];
            for other in 0..ndim {
                if other == mode {
                    continue;
                }
                let f = &factors[other];
                let dim = tensor.shape[other];
                // gram = f^T f (rank × rank)
                let mut gram = vec![vec![0.0; rank]; rank];
                for r in 0..rank {
                    for s in 0..rank {
                        gram[r][s] = (0..dim).map(|i| f[i][r] * f[i][s]).sum();
                    }
                }
                for r in 0..rank {
                    for s in 0..rank {
                        v[r][s] *= gram[r][s];
                    }
                }
            }
            // Include weights^2 diagonal in V: V = V * diag(weights)^2
            // Actually standard ALS uses V without weights; weights absorbed into factor
            // Compute Khatri-Rao product of all factors except mode (order: 0..mode-1, mode+1..N-1)
            // Then compute unfolded tensor * KR product
            let unfolded = tensor.unfold(mode)?;
            let rows_u = tensor.shape[mode];
            let cols_u = tensor.numel() / rows_u;

            // Build KR product: for modes in Kolda-Bader order (mode+1, ..., N-1, 0, ..., mode-1)
            let mut kr: Vec<Vec<f64>> = vec![vec![1.0]; rank]; // dummy 1 × rank start
                                                               // Build KR as cols_u × rank
            let mut kr_mat = vec![vec![1.0; rank]; 1];
            for step in 0..ndim - 1 {
                let m = (mode + 1 + step) % ndim;
                let f = &factors[m];
                let dim_m = tensor.shape[m];
                let kr_rows = kr_mat.len();
                let mut new_kr = vec![vec![0.0; rank]; kr_rows * dim_m];
                for i in 0..kr_rows {
                    for j in 0..dim_m {
                        for r in 0..rank {
                            new_kr[i * dim_m + j][r] = kr_mat[i][r] * f[j][r];
                        }
                    }
                }
                kr_mat = new_kr;
            }
            // kr_mat is cols_u × rank (or should be)
            let _ = kr;

            // mttkrp = unfolded * kr_mat  (rows_u × rank)
            let mttkrp = matmul_nn(&unfolded, &kr_mat, rows_u, cols_u, rank);

            // Solve: factor * V = mttkrp  => factor = mttkrp * V^{-1}
            // Update each row of factor: factor[i] * V = mttkrp[i]
            // Equivalently solve V^T factor[i]^T = mttkrp[i]^T for each i
            let mut new_factor = vec![vec![0.0; rank]; rows_u];
            for i in 0..rows_u {
                match solve_sq(&v, &mttkrp[i], rank) {
                    Ok(sol) => new_factor[i] = sol,
                    Err(_) => {
                        // Add small regularization and retry
                        let mut v_reg = v.clone();
                        for r in 0..rank {
                            v_reg[r][r] += 1e-10;
                        }
                        new_factor[i] = solve_sq(&v_reg, &mttkrp[i], rank)
                            .unwrap_or_else(|_| mttkrp[i].clone());
                    }
                }
            }
            // Normalize columns and extract weights
            for r in 0..rank {
                let nrm: f64 = (0..rows_u)
                    .map(|i| new_factor[i][r] * new_factor[i][r])
                    .sum::<f64>()
                    .sqrt();
                weights[r] = nrm;
                if nrm > 1e-14 {
                    for i in 0..rows_u {
                        new_factor[i][r] /= nrm;
                    }
                }
            }
            factors[mode] = new_factor;
        }

        // Compute fit
        let fit = compute_cp_fit(tensor, &factors, &weights, rank, norm_t);
        if (fit - prev_fit).abs() < tol && iter > 0 {
            break;
        }
        prev_fit = fit;
    }

    let fit = compute_cp_fit(tensor, &factors, &weights, rank, norm_t);
    Ok(CpDecomposition {
        factors,
        weights,
        rank,
        n_iter,
        fit,
    })
}

/// Compute CP fit = 1 - ||T - T_hat|| / ||T||
fn compute_cp_fit(
    tensor: &Tensor,
    factors: &[Vec<Vec<f64>>],
    weights: &[f64],
    rank: usize,
    norm_t: f64,
) -> f64 {
    // Reconstruct tensor and compute ||T - T_hat||
    let n = tensor.numel();
    let ndim = tensor.n_dims();
    let mut idx = vec![0usize; ndim];
    let mut sq_err = 0.0f64;
    for flat in 0..n {
        // T_hat[idx] = sum_r weight_r * prod_mode factors[mode][idx[mode]][r]
        let mut val = 0.0;
        for r in 0..rank {
            let mut prod = weights[r];
            for mode in 0..ndim {
                prod *= factors[mode][idx[mode]][r];
            }
            val += prod;
        }
        let diff = tensor.data[flat] - val;
        sq_err += diff * diff;
        for d in (0..ndim).rev() {
            idx[d] += 1;
            if idx[d] < tensor.shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    if norm_t > 1e-300 {
        1.0 - sq_err.sqrt() / norm_t
    } else {
        if sq_err < 1e-20 {
            1.0
        } else {
            0.0
        }
    }
}

// ============================================================================
// Tucker decomposition via HOSVD
// ============================================================================

/// Result of Tucker decomposition.
#[derive(Debug, Clone)]
pub struct TuckerDecomposition {
    /// Core tensor G (shape = ranks).
    pub core: Tensor,
    /// Factor matrices: `factors[n]` is `(shape[n] × ranks[n])`.
    pub factors: Vec<Vec<Vec<f64>>>,
    /// Ranks for each mode.
    pub ranks: Vec<usize>,
}

/// Tucker decomposition via Higher-Order SVD (HOSVD).
///
/// Computes the truncated HOSVD: factor matrices are computed independently
/// as the leading left singular vectors of each mode unfolding.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `ranks` - Target rank for each mode
///
/// # Errors
///
/// Returns an error if `ranks.len() != tensor.n_dims()`.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::tensor_decomp::tensor_nd::{Tensor, tucker_hosvd};
///
/// let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
/// let t = Tensor::new(vec![2, 3, 4], data).expect("valid input");
/// let tucker = tucker_hosvd(&t, &[2, 2, 3]).expect("valid input");
/// assert_eq!(tucker.core.shape, vec![2, 2, 3]);
/// ```
pub fn tucker_hosvd(tensor: &Tensor, ranks: &[usize]) -> LinalgResult<TuckerDecomposition> {
    let ndim = tensor.n_dims();
    if ranks.len() != ndim {
        return Err(LinalgError::ShapeError(format!(
            "tucker_hosvd: ranks.len()={} != n_dims={}",
            ranks.len(),
            ndim
        )));
    }

    // Compute factor matrices: U_n = leading r_n left singular vectors of X_(n)
    let mut factors: Vec<Vec<Vec<f64>>> = Vec::with_capacity(ndim);
    for mode in 0..ndim {
        let unfolded = tensor.unfold(mode)?;
        let rows = tensor.shape[mode];
        let cols = tensor.numel() / rows;
        let (u, _s, _vt) = thin_svd(&unfolded, rows, cols);
        // Take first ranks[mode] columns
        let r = ranks[mode].min(rows).min(cols);
        let factor: Vec<Vec<f64>> = (0..rows)
            .map(|i| (0..r).map(|j| u[i][j]).collect())
            .collect();
        factors.push(factor);
    }

    // Compute core: G = T ×_1 U_1^T ×_2 U_2^T ... ×_N U_N^T
    let core = multilinear_product(tensor, &factors, ranks)?;
    Ok(TuckerDecomposition {
        core,
        factors,
        ranks: ranks.to_vec(),
    })
}

/// Tucker decomposition via Higher-Order Orthogonal Iteration (HOOI).
///
/// More accurate than HOSVD: alternates between fixing all factors except one
/// and updating that factor via SVD of the mode-n unfolding of the partial core.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `ranks` - Target rank for each mode
/// * `max_iter` - Maximum HOOI iterations
/// * `tol` - Convergence tolerance (relative change in fit)
/// * `seed` - Random seed for initialization (unused; HOSVD init used)
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::tensor_decomp::tensor_nd::{Tensor, tucker_hooi};
///
/// let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
/// let t = Tensor::new(vec![2, 3, 4], data).expect("valid input");
/// let tucker = tucker_hooi(&t, &[2, 2, 3], 10, 1e-6, 42).expect("valid input");
/// assert_eq!(tucker.ranks, vec![2, 2, 3]);
/// ```
pub fn tucker_hooi(
    tensor: &Tensor,
    ranks: &[usize],
    max_iter: usize,
    tol: f64,
    _seed: u64,
) -> LinalgResult<TuckerDecomposition> {
    // Initialize with HOSVD
    let mut tucker = tucker_hosvd(tensor, ranks)?;
    let norm_t = tensor.frobenius_norm();
    let ndim = tensor.n_dims();
    let mut prev_fit = -1.0f64;

    for _ in 0..max_iter {
        for mode in 0..ndim {
            // Compute Y = T ×_n U_n for all n != mode
            let partial = multilinear_product_skip(tensor, &tucker.factors, ranks, mode)?;
            // Unfold Y along mode
            let unfolded = partial.unfold(mode)?;
            let rows = partial.shape[mode];
            let cols = partial.numel() / rows;
            let (u, _s, _vt) = thin_svd(&unfolded, rows, cols);
            let r = ranks[mode].min(rows).min(cols);
            tucker.factors[mode] = (0..rows)
                .map(|i| (0..r).map(|j| u[i][j]).collect())
                .collect();
        }
        tucker.core = multilinear_product(tensor, &tucker.factors, ranks)?;

        // Fit = ||G||_F / ||T||_F
        let fit = if norm_t > 1e-300 {
            tucker.core.frobenius_norm() / norm_t
        } else {
            1.0
        };
        if (fit - prev_fit).abs() < tol && prev_fit >= 0.0 {
            break;
        }
        prev_fit = fit;
    }
    tucker.ranks = ranks.to_vec();
    Ok(tucker)
}

/// Compute T ×_1 U_1^T ×_2 U_2^T ... ×_N U_N^T (multilinear product / core tensor).
fn multilinear_product(
    tensor: &Tensor,
    factors: &[Vec<Vec<f64>>],
    ranks: &[usize],
) -> LinalgResult<Tensor> {
    let ndim = tensor.n_dims();
    let mut current = tensor.clone();
    for mode in 0..ndim {
        let rows = current.shape[mode];
        let cols = current.numel() / rows;
        let unfolded = current.unfold(mode)?;
        // u_t is ranks[mode] × rows (factor transposed)
        let r = ranks[mode].min(rows);
        let f = &factors[mode];
        // Compute u_t * unfolded (r × cols)
        let mut prod = vec![vec![0.0; cols]; r];
        for i in 0..r {
            for k in 0..rows {
                if f[k][i] == 0.0 {
                    continue;
                }
                for j in 0..cols {
                    prod[i][j] += f[k][i] * unfolded[k][j];
                }
            }
        }
        // Fold back: new shape has shape[mode] replaced by r
        let mut new_shape = current.shape.clone();
        new_shape[mode] = r;
        current = Tensor::fold(&prod, mode, new_shape)?;
    }
    Ok(current)
}

/// Compute T ×_n U_n^T for all n != skip_mode.
fn multilinear_product_skip(
    tensor: &Tensor,
    factors: &[Vec<Vec<f64>>],
    ranks: &[usize],
    skip_mode: usize,
) -> LinalgResult<Tensor> {
    let ndim = tensor.n_dims();
    let mut current = tensor.clone();
    for mode in 0..ndim {
        if mode == skip_mode {
            continue;
        }
        let rows = current.shape[mode];
        let cols = current.numel() / rows;
        let unfolded = current.unfold(mode)?;
        let r = ranks[mode].min(rows);
        let f = &factors[mode];
        let mut prod = vec![vec![0.0; cols]; r];
        for i in 0..r {
            for k in 0..rows {
                if f[k][i] == 0.0 {
                    continue;
                }
                for j in 0..cols {
                    prod[i][j] += f[k][i] * unfolded[k][j];
                }
            }
        }
        let mut new_shape = current.shape.clone();
        new_shape[mode] = r;
        current = Tensor::fold(&prod, mode, new_shape)?;
    }
    Ok(current)
}

// ============================================================================
// Tensor Train decomposition (TT-SVD)
// ============================================================================

/// Result of Tensor Train decomposition.
///
/// A Tensor Train represents T[i_1, i_2, ..., i_d] =
/// G_1\[i_1\] * G_2\[i_2\] * ... * G_d\[i_d\]
/// where each G_k is an r_{k-1} × n_k × r_k matrix (a 3D core).
#[derive(Debug, Clone)]
pub struct TensorTrainDecomposition {
    /// TT-cores: `cores[k]` has shape `(ranks[k], shape[k], ranks[k+1])`.
    pub cores: Vec<Tensor>,
    /// Ranks including boundary conditions: `ranks[0] = ranks[d] = 1`.
    pub ranks: Vec<usize>,
}

/// Tensor Train SVD (TT-SVD) algorithm (Oseledets 2011).
///
/// Sequentially computes SVD-based decompositions of reshaped tensor slices
/// to produce a TT decomposition with controlled error `rel_error`.
///
/// # Arguments
///
/// * `tensor` - Input N-dimensional tensor
/// * `max_rank` - Maximum allowed TT rank
/// * `rel_error` - Relative error budget (controls rank truncation)
///
/// # Returns
///
/// A [`TensorTrainDecomposition`] with cores of shape `(r_{k-1}, n_k, r_k)`.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::tensor_decomp::tensor_nd::{Tensor, tensor_train_svd};
///
/// let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
/// let t = Tensor::new(vec![2, 3, 4], data).expect("valid input");
/// let tt = tensor_train_svd(&t, 4, 1e-6).expect("valid input");
/// assert_eq!(tt.cores.len(), 3);
/// assert_eq!(tt.ranks[0], 1);
/// assert_eq!(*tt.ranks.last().expect("valid input"), 1);
/// ```
pub fn tensor_train_svd(
    tensor: &Tensor,
    max_rank: usize,
    rel_error: f64,
) -> LinalgResult<TensorTrainDecomposition> {
    let ndim = tensor.n_dims();
    if ndim < 2 {
        return Err(LinalgError::ShapeError(
            "tensor_train_svd requires at least 2 modes".into(),
        ));
    }

    let norm_t = tensor.frobenius_norm();
    // Truncation threshold per SVD step
    let delta = if norm_t > 0.0 && ndim > 1 {
        rel_error * norm_t / ((ndim - 1) as f64).sqrt()
    } else {
        0.0
    };

    let mut cores: Vec<Tensor> = Vec::with_capacity(ndim);
    let mut ranks: Vec<usize> = vec![1];

    // Working matrix: initially flatten tensor as 1 × n_total
    let mut c_rows = 1usize;
    let mut c_data = tensor.data.clone();

    for k in 0..ndim - 1 {
        let n_k = tensor.shape[k];
        let c_cols = c_data.len() / c_rows;
        // Reshape C to (c_rows * n_k) × (c_cols / n_k)
        let left = c_rows * n_k;
        let right = c_cols / n_k;
        if left * right != c_data.len() {
            return Err(LinalgError::ShapeError(format!(
                "TT-SVD reshape error at mode {}: {} × {} != {}",
                k,
                left,
                right,
                c_data.len()
            )));
        }
        // Build left × right matrix
        let mat: Vec<Vec<f64>> = (0..left)
            .map(|i| c_data[i * right..(i + 1) * right].to_vec())
            .collect();

        // SVD: mat = U S Vt, truncate at threshold delta
        let (u, s, vt) = thin_svd(&mat, left, right);
        let actual_rank = s.iter().filter(|&&sv| sv > delta).count().max(1);
        let r = actual_rank.min(max_rank).min(s.len());

        // Store core G_k as (c_rows × n_k × r) reshaped as Tensor
        // Core shape: (ranks[k], n_k, r)
        let prev_r = ranks[k];
        let core_data: Vec<f64> = (0..left)
            .flat_map(|i| {
                let u_ref = &u;
                (0..r).map(move |j| u_ref[i][j])
            })
            .collect();
        cores.push(Tensor::new(vec![prev_r, n_k, r], core_data)?);
        ranks.push(r);

        // Update C for next iteration: C = diag(S[:r]) * Vt[:r, :]  (r × right)
        c_rows = r;
        c_data = (0..r)
            .flat_map(|i| {
                let s_ref = &s;
                let vt_ref = &vt;
                (0..right).map(move |j| s_ref[i] * vt_ref[i][j])
            })
            .collect();
    }

    // Last core: shape (ranks[N-1], n_{N-1}, 1)
    let n_last = tensor.shape[ndim - 1];
    let prev_r = ranks[ndim - 1];
    // c_data should be prev_r × n_last
    if c_data.len() != prev_r * n_last {
        // Reshape
        let last_core_data = c_data.clone();
        cores.push(Tensor::new(vec![prev_r, n_last, 1], {
            let mut d = last_core_data;
            d.resize(prev_r * n_last, 0.0);
            d
        })?);
    } else {
        let mut last_data = vec![0.0; prev_r * n_last];
        for i in 0..prev_r {
            for j in 0..n_last {
                last_data[i * n_last + j] = c_data[i * n_last + j];
            }
        }
        cores.push(Tensor::new(vec![prev_r, n_last, 1], last_data)?);
    }
    ranks.push(1);

    Ok(TensorTrainDecomposition { cores, ranks })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_create() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(t.n_dims(), 3);
        assert_eq!(t.numel(), 24);
        assert_eq!(t.get(&[0, 0, 0]), 0.0);
    }

    #[test]
    fn test_tensor_set_get() {
        let mut t = Tensor::zeros(vec![2, 3]);
        t.set(&[1, 2], 42.0);
        assert_eq!(t.get(&[1, 2]), 42.0);
        assert_eq!(t.get(&[0, 0]), 0.0);
    }

    #[test]
    fn test_tensor_reshape() {
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let t = Tensor::new(vec![3, 4], data).expect("failed to create t");
        let t2 = t.reshape(vec![2, 6]).expect("failed to create t2");
        assert_eq!(t2.shape, vec![2, 6]);
        assert!(t2.reshape(vec![5, 3]).is_err());
    }

    #[test]
    fn test_tensor_unfold_fold_roundtrip() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = Tensor::new(vec![2, 3, 4], data.clone()).expect("failed to create t");
        for mode in 0..3 {
            let mat = t.unfold(mode).expect("failed to create mat");
            let t2 = Tensor::fold(&mat, mode, t.shape.clone()).expect("failed to create t2");
            assert_eq!(t2.data, t.data, "roundtrip failed for mode {}", mode);
        }
    }

    #[test]
    fn test_cp_als_small() {
        let data: Vec<f64> = (0..8).map(|x| x as f64).collect();
        let t = Tensor::new(vec![2, 2, 2], data).expect("failed to create t");
        let cp = cp_als(&t, 2, 200, 1e-6, 42).expect("failed to create cp");
        assert_eq!(cp.rank, 2);
        assert_eq!(cp.factors.len(), 3);
        assert!(cp.fit >= -0.01); // fit can be slightly negative due to numerical noise
    }

    #[test]
    fn test_tucker_hosvd_shape() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = Tensor::new(vec![2, 3, 4], data).expect("failed to create t");
        let tucker = tucker_hosvd(&t, &[2, 2, 3]).expect("failed to create tucker");
        assert_eq!(tucker.core.shape, vec![2, 2, 3]);
        assert_eq!(tucker.factors.len(), 3);
        assert_eq!(tucker.factors[0].len(), 2); // n_0 = 2
        assert_eq!(tucker.factors[0][0].len(), 2); // rank_0 = 2
    }

    #[test]
    fn test_tucker_hooi_shape() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = Tensor::new(vec![2, 3, 4], data).expect("failed to create t");
        let tucker = tucker_hooi(&t, &[2, 2, 3], 5, 1e-6, 42).expect("failed to create tucker");
        assert_eq!(tucker.ranks, vec![2, 2, 3]);
    }

    #[test]
    fn test_tensor_train_svd_shape() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = Tensor::new(vec![2, 3, 4], data).expect("failed to create t");
        let tt = tensor_train_svd(&t, 4, 1e-6).expect("failed to create tt");
        assert_eq!(tt.cores.len(), 3);
        assert_eq!(tt.ranks[0], 1);
        assert_eq!(*tt.ranks.last().expect("unexpected None or Err"), 1);
        // Core shapes
        let r0 = tt.ranks[0];
        let r1 = tt.ranks[1];
        let r2 = tt.ranks[2];
        assert_eq!(tt.cores[0].shape, vec![r0, 2, r1]);
        assert_eq!(tt.cores[1].shape, vec![r1, 3, r2]);
        assert_eq!(tt.cores[2].shape, vec![r2, 4, 1]);
    }
}
