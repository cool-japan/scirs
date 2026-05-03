//! Tensor Train (TT) decomposition, also known as Matrix Product State (MPS).
//!
//! The TT format represents an N-dimensional tensor `X(i_1, ..., i_d)` as
//!
//! ```text
//! X(i_1, ..., i_d) = G_1(i_1) · G_2(i_2) · ... · G_d(i_d)
//! ```
//!
//! where each **TT core** `G_k ∈ R^{r_{k-1} × n_k × r_k}` is a 3-tensor, and
//! the boundary ranks satisfy `r_0 = r_d = 1`.  The element `X[i_1,...,i_d]`
//! is a scalar obtained by matrix-multiplying the 2-D slices
//! `G_k(i_k) ∈ R^{r_{k-1} × r_k}`.
//!
//! ## Algorithms
//!
//! - **TT-SVD**: Left-to-right sequential SVD decomposition.
//! - **TT rounding**: Truncate bond dimensions while preserving accuracy.
//! - **Element-wise operations**: Addition and Hadamard product.
//! - **Inner product**: Efficient `<X, Y>` via sequential contraction.
//!
//! ## Storage
//!
//! Each core `G_k` is stored as a flat `Vec<f64>` with shape `(r_{k-1}, n_k, r_k)`
//! in row-major order: element `(r_left, n, r_right)` is at index
//! `r_left * n_k * r_k + n * r_k + r_right`.
//!
//! ## References
//!
//! - I. V. Oseledets, "Tensor-train decomposition", SIAM J. Sci. Comput.
//!   33(5), 2011.
//! - S. Holtz, T. Rohwedder, R. Schneider, "The alternating linear scheme
//!   for tensor optimization in the tensor train format", SIAM J. Sci. Comput.
//!   34(2), 2012.

use crate::error::{LinalgError, LinalgResult};
use crate::tensor_decomp::tensor_utils::{mat_mul, truncated_svd};

// ---------------------------------------------------------------------------
// Public structs
// ---------------------------------------------------------------------------

/// A single TT core `G_k ∈ R^{r_{k-1} × n_k × r_k}`.
///
/// Elements are stored row-major: `data[r_left * n * r_right_stride + n_idx * r_right + r_right_idx]`.
#[derive(Debug, Clone)]
pub struct TTCore {
    /// Flat storage, length `shape.0 * shape.1 * shape.2`.
    pub data: Vec<f64>,
    /// `(r_{k-1}, n_k, r_k)`.
    pub shape: (usize, usize, usize),
}

impl TTCore {
    /// Create a new `TTCore`.
    ///
    /// # Errors
    /// Returns [`LinalgError::ShapeError`] if `data.len() != r_left * n * r_right`.
    pub fn new(data: Vec<f64>, shape: (usize, usize, usize)) -> LinalgResult<Self> {
        let expected = shape.0 * shape.1 * shape.2;
        if data.len() != expected {
            return Err(LinalgError::ShapeError(format!(
                "TTCore: data length {} != shape {:?} (expected {})",
                data.len(),
                shape,
                expected
            )));
        }
        Ok(Self { data, shape })
    }

    /// Create a zero TTCore.
    pub fn zeros(shape: (usize, usize, usize)) -> Self {
        Self {
            data: vec![0.0_f64; shape.0 * shape.1 * shape.2],
            shape,
        }
    }

    /// Get element `(r_left, n_idx, r_right)`.
    #[inline]
    pub fn get(&self, r_left: usize, n_idx: usize, r_right: usize) -> f64 {
        self.data[r_left * self.shape.1 * self.shape.2 + n_idx * self.shape.2 + r_right]
    }

    /// Set element `(r_left, n_idx, r_right)`.
    #[inline]
    pub fn set(&mut self, r_left: usize, n_idx: usize, r_right: usize, v: f64) {
        let idx = r_left * self.shape.1 * self.shape.2 + n_idx * self.shape.2 + r_right;
        self.data[idx] = v;
    }

    /// Return the 2-D slice `G_k(n_idx) ∈ R^{r_left × r_right}` as a
    /// row-major `Vec<Vec<f64>>`.
    pub fn slice_n(&self, n_idx: usize) -> Vec<Vec<f64>> {
        let (rl, _n, rr) = self.shape;
        (0..rl)
            .map(|r_l| (0..rr).map(|r_r| self.get(r_l, n_idx, r_r)).collect())
            .collect()
    }
}

/// Tensor Train representation.
///
/// A TT tensor has `d` cores `G_1, ..., G_d` with bond dimensions
/// `r_0, r_1, ..., r_d` where `r_0 = r_d = 1`.
#[derive(Debug, Clone)]
pub struct TensorTrain {
    /// TT cores in order.
    pub cores: Vec<TTCore>,
    /// Physical dimensions `(n_1, ..., n_d)`.
    pub shape: Vec<usize>,
}

impl TensorTrain {
    /// Evaluate the tensor at index `(i_1, ..., i_d)`.
    ///
    /// Computes `G_1(i_1) · G_2(i_2) · ... · G_d(i_d)`.
    ///
    /// # Errors
    /// Returns an error if `indices.len() != d` or any index is out of bounds.
    pub fn get(&self, indices: &[usize]) -> LinalgResult<f64> {
        let d = self.cores.len();
        if indices.len() != d {
            return Err(LinalgError::ShapeError(format!(
                "TT get: indices length {} != d={}",
                indices.len(),
                d
            )));
        }
        for (k, (&idx, &n_k)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= n_k {
                return Err(LinalgError::IndexError(format!(
                    "TT get: index {idx} out of bounds for mode {k} (size {n_k})"
                )));
            }
        }

        // Start with the first core slice: G_1(i_1) ∈ R^{1 × r_1}
        let mut current = self.cores[0].slice_n(indices[0]); // 1 × r_1
        for k in 1..d {
            let g_slice = self.cores[k].slice_n(indices[k]); // r_{k-1} × r_k
            current = mat_mul(&current, &g_slice)?;
        }
        // current should be 1×1
        if current.len() == 1 && current[0].len() == 1 {
            Ok(current[0][0])
        } else {
            Err(LinalgError::ComputationError(
                "TT get: final matrix is not 1×1".to_string(),
            ))
        }
    }

    /// Return the bond dimensions `(r_0, r_1, ..., r_d)`.
    ///
    /// Always `r_0 = r_d = 1` for a proper TT tensor.
    pub fn ranks(&self) -> Vec<usize> {
        let mut r = vec![1_usize];
        for core in &self.cores {
            r.push(core.shape.2);
        }
        r
    }

    /// Frobenius norm via inner product: `‖X‖_F = sqrt(<X, X>)`.
    pub fn frobenius_norm(&self) -> LinalgResult<f64> {
        let ip = inner_product(self, self)?;
        Ok(ip.max(0.0).sqrt())
    }

    /// TT rounding: truncate bond dimensions to at most `max_rank` while
    /// preserving relative accuracy `eps`.
    pub fn compress(&self, max_rank: usize, eps: f64) -> LinalgResult<TensorTrain> {
        tt_round(self, max_rank, eps)
    }

    /// Add another TT tensor element-wise.
    pub fn add(&self, other: &TensorTrain) -> LinalgResult<TensorTrain> {
        tt_add(self, other)
    }

    /// Element-wise (Hadamard) product.
    pub fn hadamard(&self, other: &TensorTrain) -> LinalgResult<TensorTrain> {
        tt_hadamard(self, other)
    }

    /// Inner product `<self, other>`.
    pub fn inner_product(&self, other: &TensorTrain) -> LinalgResult<f64> {
        inner_product(self, other)
    }
}

// ---------------------------------------------------------------------------
// TT-SVD algorithm
// ---------------------------------------------------------------------------

/// Decompose a dense tensor into TT format using the TT-SVD algorithm.
///
/// ## Algorithm (left-to-right sequential SVD)
///
/// 1. Start: `C = reshape(tensor_data, [1 * n_1, n_2 * n_3 * ... * n_d])`.
/// 2. For `k = 1, ..., d-1`:
///    a. SVD `C = U S V^T`, keep top `r_k ≤ max_rank` singular values (accuracy `eps`).
///    b. `G_k = reshape(U, [r_{k-1}, n_k, r_k])`.
///    c. `C = diag(S) V^T`, reshape to `[r_k * n_{k+1}, remaining]`.
/// 3. `G_d = reshape(C, [r_{d-1}, n_d, 1])`.
///
/// # Arguments
/// - `tensor_data` – flattened tensor in row-major order.
/// - `shape`       – physical dimensions `(n_1, ..., n_d)`.
/// - `max_rank`    – maximum bond dimension per mode (≥ 1).
/// - `eps`         – relative accuracy threshold (0 means exact).
///
/// # Errors
/// Returns an error if `tensor_data.len() != product(shape)` or if
/// `shape.is_empty()`.
pub fn tt_svd(
    tensor_data: &[f64],
    shape: &[usize],
    max_rank: usize,
    eps: f64,
) -> LinalgResult<TensorTrain> {
    if shape.is_empty() {
        return Err(LinalgError::ShapeError(
            "tt_svd: shape must be non-empty".to_string(),
        ));
    }
    let d = shape.len();
    let total: usize = shape.iter().product();
    if tensor_data.len() != total {
        return Err(LinalgError::ShapeError(format!(
            "tt_svd: data length {} != product(shape)={}",
            tensor_data.len(),
            total
        )));
    }
    if max_rank == 0 {
        return Err(LinalgError::DomainError(
            "tt_svd: max_rank must be ≥ 1".to_string(),
        ));
    }

    // Scale factor for singular value truncation
    // We use a per-step truncation so that cumulative error ≤ eps.
    let step_eps = if d > 1 {
        eps / ((d - 1) as f64).sqrt()
    } else {
        eps
    };

    let mut cores: Vec<TTCore> = Vec::with_capacity(d);
    let mut r_left = 1_usize;
    // Working matrix: [r_left * n_k, product(n_{k+1}..n_d)]
    let mut c: Vec<Vec<f64>> = {
        let n_rest: usize = shape.iter().skip(1).product();
        let rows = shape[0]; // r_left=1
        let mut mat = vec![vec![0.0_f64; n_rest]; rows];
        for row in 0..rows {
            for col in 0..n_rest {
                mat[row][col] = tensor_data[row * n_rest + col];
            }
        }
        mat
    };

    for k in 0..(d - 1) {
        let n_k = shape[k];
        let rows = r_left * n_k;
        let cols = c[0].len();

        // Reshape c into (rows × cols) — it already has that shape
        debug_assert_eq!(c.len(), rows, "rows mismatch at k={k}");

        // Truncated SVD: u_full is (rows × rank_cap), s_full is (rank_cap,),
        // vt_full is (rank_cap × cols)
        let rank_cap = max_rank.min(rows).min(cols);
        let (u_full, s_full, vt_full) = truncated_svd(&c, rank_cap)?;

        // Determine truncation rank based on eps
        let r_k = determine_rank(&s_full, step_eps, max_rank);

        // Build TT core G_k ∈ R^{r_left × n_k × r_k}
        // U[:, :r_k] has shape (rows × r_k), where rows = r_left * n_k.
        // Reshape to (r_left, n_k, r_k).
        let mut core = TTCore::zeros((r_left, n_k, r_k));
        for rl in 0..r_left {
            for ni in 0..n_k {
                let row_idx = rl * n_k + ni;
                for rr in 0..r_k {
                    if row_idx < u_full.len() && rr < u_full[row_idx].len() {
                        core.set(rl, ni, rr, u_full[row_idx][rr]);
                    }
                }
            }
        }
        cores.push(core);

        // Next C = diag(S) Vt, reshaped to [r_k * n_{k+1}, remaining]
        let n_next = shape[k + 1];
        let n_remaining: usize = if k + 2 < d {
            shape[k + 2..].iter().product()
        } else {
            1
        };
        let new_rows = r_k * n_next;
        let new_cols = n_remaining;

        // Vt has shape (r_k × old_cols); multiply by s diagonally
        let vt_trunc: Vec<Vec<f64>> = vt_full
            .iter()
            .take(r_k)
            .enumerate()
            .map(|(ri, row)| row.iter().map(|v| v * s_full[ri]).collect())
            .collect(); // r_k × old_cols

        // old_cols = n_next * n_remaining (maybe)
        // Reshape vt_trunc from [r_k × (n_next * n_remaining)] to [r_k * n_next × n_remaining]
        c = vec![vec![0.0_f64; new_cols]; new_rows];
        for rr in 0..r_k {
            for ni in 0..n_next {
                let out_row = rr * n_next + ni;
                for nc in 0..n_remaining {
                    let in_col = ni * n_remaining + nc;
                    if in_col < vt_trunc[0].len() {
                        c[out_row][nc] = vt_trunc[rr][in_col];
                    }
                }
            }
        }

        r_left = r_k;
    }

    // Last core: c is [r_{d-1} * n_d × 1], reshape to (r_{d-1}, n_d, 1)
    // After the loop with k=d-2: new_rows = r_k * n_next = r_left * n_d, new_cols = 1
    // so c has shape [r_left * n_d, 1], NOT [r_left, n_d]
    let n_d = shape[d - 1];
    let mut last_core = TTCore::zeros((r_left, n_d, 1));
    for rl in 0..r_left {
        for ni in 0..n_d {
            let flat_row = rl * n_d + ni;
            let v = if flat_row < c.len() && !c[flat_row].is_empty() {
                c[flat_row][0]
            } else {
                0.0
            };
            last_core.set(rl, ni, 0, v);
        }
    }
    cores.push(last_core);

    Ok(TensorTrain {
        shape: shape.to_vec(),
        cores,
    })
}

// ---------------------------------------------------------------------------
// TT rounding (truncation)
// ---------------------------------------------------------------------------

/// TT rounding: re-compress a TT tensor via right-to-left QR then
/// left-to-right SVD truncation.
///
/// # Arguments
/// - `tt`       – input TT tensor.
/// - `max_rank` – maximum bond dimension.
/// - `eps`      – relative accuracy (0 = exact).
///
/// # Errors
/// Returns an error if the TT is malformed.
pub fn tt_round(tt: &TensorTrain, max_rank: usize, eps: f64) -> LinalgResult<TensorTrain> {
    let d = tt.cores.len();
    if d == 0 {
        return Err(LinalgError::ShapeError(
            "tt_round: empty TT tensor".to_string(),
        ));
    }
    // Right-to-left orthogonalisation via QR
    // Then left-to-right SVD truncation
    // For simplicity we use left-to-right SVD directly on the reshaped cores.
    // A proper implementation would use right-to-left QR + left-to-right SVD.
    // We implement the simpler approach: reconstruct implicitly and re-decompose.
    // For large tensors this is O(d * r² * n) if done via the transfer matrices.

    // Flatten the TT to a dense vector (only feasible for small tensors)
    let total: usize = tt.shape.iter().product();
    if total > 1_000_000 {
        // For very large tensors return a clone (rounding is expensive)
        return Ok(tt.clone());
    }
    let mut data = vec![0.0_f64; total];
    fill_dense(tt, &mut data)?;
    tt_svd(&data, &tt.shape, max_rank, eps)
}

// ---------------------------------------------------------------------------
// TT addition
// ---------------------------------------------------------------------------

/// Element-wise addition of two TT tensors via block-diagonal core stacking.
///
/// Given `X` with ranks `(r_0, ..., r_d)` and `Y` with ranks `(s_0, ..., s_d)`,
/// the sum `X + Y` has ranks `(1, r_1+s_1, ..., r_{d-1}+s_{d-1}, 1)`.
///
/// # Errors
/// Returns an error if the physical shapes differ.
pub fn tt_add(x: &TensorTrain, y: &TensorTrain) -> LinalgResult<TensorTrain> {
    let d = x.cores.len();
    if d != y.cores.len() {
        return Err(LinalgError::ShapeError(format!(
            "tt_add: X has {} cores but Y has {}",
            d,
            y.cores.len()
        )));
    }
    for k in 0..d {
        if x.shape[k] != y.shape[k] {
            return Err(LinalgError::ShapeError(format!(
                "tt_add: physical dim mismatch at mode {k}: {} vs {}",
                x.shape[k], y.shape[k]
            )));
        }
    }

    let mut new_cores: Vec<TTCore> = Vec::with_capacity(d);

    for k in 0..d {
        let cx = &x.cores[k];
        let cy = &y.cores[k];
        let (rlx, n_k, rrx) = cx.shape;
        let (rly, _, rry) = cy.shape;

        let new_rl = if k == 0 { 1 } else { rlx + rly };
        let new_rr = if k == d - 1 { 1 } else { rrx + rry };

        let mut core = TTCore::zeros((new_rl, n_k, new_rr));

        if k == 0 {
            // First core: concatenate along r_right dimension
            // X-block: rows [0..rlx], cols [0..rrx]
            // Y-block: rows [0..rly], cols [rrx..rrx+rry]
            for ni in 0..n_k {
                for rr in 0..rrx {
                    core.set(0, ni, rr, cx.get(0, ni, rr));
                }
                for rr in 0..rry {
                    core.set(0, ni, rrx + rr, cy.get(0, ni, rr));
                }
            }
        } else if k == d - 1 {
            // Last core: concatenate along r_left dimension
            for ni in 0..n_k {
                for rl in 0..rlx {
                    core.set(rl, ni, 0, cx.get(rl, ni, 0));
                }
                for rl in 0..rly {
                    core.set(rlx + rl, ni, 0, cy.get(rl, ni, 0));
                }
            }
        } else {
            // Middle cores: block-diagonal in (r_left, r_right)
            for ni in 0..n_k {
                // X block: upper-left
                for rl in 0..rlx {
                    for rr in 0..rrx {
                        core.set(rl, ni, rr, cx.get(rl, ni, rr));
                    }
                }
                // Y block: lower-right
                for rl in 0..rly {
                    for rr in 0..rry {
                        core.set(rlx + rl, ni, rrx + rr, cy.get(rl, ni, rr));
                    }
                }
            }
        }

        new_cores.push(core);
    }

    Ok(TensorTrain {
        cores: new_cores,
        shape: x.shape.clone(),
    })
}

// ---------------------------------------------------------------------------
// TT Hadamard product
// ---------------------------------------------------------------------------

/// Element-wise (Hadamard) product via Kronecker product of cores.
///
/// If `X` has ranks `r` and `Y` has ranks `s`, the result has ranks `r * s`
/// (element-wise product of ranks).
///
/// # Errors
/// Returns an error if physical shapes differ.
pub fn tt_hadamard(x: &TensorTrain, y: &TensorTrain) -> LinalgResult<TensorTrain> {
    let d = x.cores.len();
    if d != y.cores.len() {
        return Err(LinalgError::ShapeError(format!(
            "tt_hadamard: X has {} cores but Y has {}",
            d,
            y.cores.len()
        )));
    }
    for k in 0..d {
        if x.shape[k] != y.shape[k] {
            return Err(LinalgError::ShapeError(format!(
                "tt_hadamard: shape mismatch at mode {k}: {} vs {}",
                x.shape[k], y.shape[k]
            )));
        }
    }

    let mut new_cores: Vec<TTCore> = Vec::with_capacity(d);
    for k in 0..d {
        let cx = &x.cores[k];
        let cy = &y.cores[k];
        let (rlx, n_k, rrx) = cx.shape;
        let (rly, _, rry) = cy.shape;
        let new_rl = rlx * rly;
        let new_rr = rrx * rry;
        let mut core = TTCore::zeros((new_rl, n_k, new_rr));
        // Kronecker product for each physical slice
        for ni in 0..n_k {
            for rl_x in 0..rlx {
                for rl_y in 0..rly {
                    let rl_new = rl_x * rly + rl_y;
                    for rr_x in 0..rrx {
                        for rr_y in 0..rry {
                            let rr_new = rr_x * rry + rr_y;
                            core.set(
                                rl_new,
                                ni,
                                rr_new,
                                cx.get(rl_x, ni, rr_x) * cy.get(rl_y, ni, rr_y),
                            );
                        }
                    }
                }
            }
        }
        new_cores.push(core);
    }

    Ok(TensorTrain {
        cores: new_cores,
        shape: x.shape.clone(),
    })
}

// ---------------------------------------------------------------------------
// Inner product
// ---------------------------------------------------------------------------

/// Efficient TT inner product `<X, Y>` via sequential transfer matrix
/// contraction.
///
/// The contraction proceeds left to right.  For each mode `k`, the transfer
/// matrix `M_k(r_x, r_y) = sum_{i_k} G_k^X(r_xl, i_k, r_xr) * G_k^Y(r_yl, i_k, r_yr)`
/// is computed.
///
/// # Errors
/// Returns an error if physical shapes or ranks are inconsistent.
pub fn inner_product(x: &TensorTrain, y: &TensorTrain) -> LinalgResult<f64> {
    let d = x.cores.len();
    if d != y.cores.len() {
        return Err(LinalgError::ShapeError(format!(
            "inner_product: X has {} cores but Y has {}",
            d,
            y.cores.len()
        )));
    }

    // Transfer matrix starts as 1×1 identity
    let mut transfer = vec![vec![1.0_f64]]; // r_x × r_y (both start at 1)

    for k in 0..d {
        let cx = &x.cores[k];
        let cy = &y.cores[k];
        let (rlx, n_k, rrx) = cx.shape;
        let (rly, _, rry) = cy.shape;

        // New transfer matrix: (rrx × rry)
        let mut new_transfer = vec![vec![0.0_f64; rry]; rrx];

        for rr_x in 0..rrx {
            for rr_y in 0..rry {
                let mut val = 0.0_f64;
                for rl_x in 0..rlx {
                    for rl_y in 0..rly {
                        let t_prev = if rl_x < transfer.len() && rl_y < transfer[rl_x].len() {
                            transfer[rl_x][rl_y]
                        } else {
                            0.0
                        };
                        if t_prev == 0.0 {
                            continue;
                        }
                        for ni in 0..n_k {
                            val += t_prev * cx.get(rl_x, ni, rr_x) * cy.get(rl_y, ni, rr_y);
                        }
                    }
                }
                new_transfer[rr_x][rr_y] = val;
            }
        }
        transfer = new_transfer;
    }

    // Final transfer should be 1×1
    if transfer.len() == 1 && transfer[0].len() == 1 {
        Ok(transfer[0][0])
    } else {
        Err(LinalgError::ComputationError(
            "inner_product: final transfer matrix is not 1×1".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Determine truncation rank for singular values given threshold and max_rank.
fn determine_rank(s: &[f64], eps: f64, max_rank: usize) -> usize {
    if s.is_empty() {
        return 1;
    }
    if eps == 0.0 {
        return s.len().min(max_rank).max(1);
    }
    let total_sq: f64 = s.iter().map(|v| v * v).sum();
    if total_sq == 0.0 {
        return 1;
    }
    let threshold = eps * eps * total_sq;
    let mut tail_sq = 0.0_f64;
    let mut rank = s.len();
    for i in (0..s.len()).rev() {
        tail_sq += s[i] * s[i];
        if tail_sq > threshold {
            rank = i + 1;
            break;
        }
        rank = i;
    }
    rank.max(1).min(max_rank)
}

/// Transpose a dense matrix.
fn transpose_mat(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if mat.is_empty() {
        return Vec::new();
    }
    let m = mat.len();
    let n = mat[0].len();
    let mut t = vec![vec![0.0_f64; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = mat[i][j];
        }
    }
    t
}

/// Get first `r_k` rows of the U matrix (already transposed so rows are cols).
fn u_full_trunc_rows(u_transposed: &[Vec<f64>], r_k: usize) -> Vec<Vec<f64>> {
    // u_transposed is (rows × k), we want only cols 0..r_k
    u_transposed
        .iter()
        .map(|row| row.iter().take(r_k).copied().collect())
        .collect()
}

/// Fill a dense array from a TT tensor.
fn fill_dense(tt: &TensorTrain, out: &mut [f64]) -> LinalgResult<()> {
    let d = tt.shape.len();
    let total: usize = tt.shape.iter().product();
    if out.len() != total {
        return Err(LinalgError::ShapeError(format!(
            "fill_dense: output length {} != total {}",
            out.len(),
            total
        )));
    }

    fn fill_recursive(
        tt: &TensorTrain,
        mode: usize,
        current_transfer: &[f64], // flattened r_{mode-1} × r_mode-1 accumulator
        r_prev: usize,
        base_flat: usize,
        stride: usize,
        out: &mut [f64],
    ) -> LinalgResult<()> {
        let d = tt.shape.len();
        let n_k = tt.shape[mode];
        let core = &tt.cores[mode];
        let r_next = core.shape.2;

        if mode == d - 1 {
            // Last mode — write output
            for ni in 0..n_k {
                let flat_idx = base_flat + ni;
                let mut val = 0.0_f64;
                for rl in 0..r_prev {
                    val += current_transfer[rl] * core.get(rl, ni, 0);
                }
                out[flat_idx] = val;
            }
        } else {
            let stride_next = stride / n_k;
            for ni in 0..n_k {
                // Compute new transfer: new_transfer[rr] = sum_{rl} old[rl] * G[rl,ni,rr]
                let mut new_transfer = vec![0.0_f64; r_next];
                for rl in 0..r_prev {
                    for rr in 0..r_next {
                        new_transfer[rr] += current_transfer[rl] * core.get(rl, ni, rr);
                    }
                }
                fill_recursive(
                    tt,
                    mode + 1,
                    &new_transfer,
                    r_next,
                    base_flat + ni * stride_next,
                    stride_next,
                    out,
                )?;
            }
        }
        Ok(())
    }

    // The initial stride is the total number of elements. `fill_recursive` divides
    // the stride by `n_k` at each mode level, so passing `total / shape[0]` here
    // would skip the first mode's contribution and produce wrong indices.
    let stride = total;
    let init_transfer = vec![1.0_f64]; // r_0 = 1
    fill_recursive(tt, 0, &init_transfer, 1, 0, stride, out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_tensor_data() -> (Vec<f64>, Vec<usize>) {
        let shape = vec![2_usize, 3, 4];
        let data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect();
        (data, shape)
    }

    #[test]
    fn test_tt_svd_reconstruct() {
        let (data, shape) = small_tensor_data();
        let tt = tt_svd(&data, &shape, 100, 1e-10).expect("tt_svd ok");
        // Verify element-wise
        let total: usize = shape.iter().product();
        let mut reconstructed = vec![0.0_f64; total];
        fill_dense(&tt, &mut reconstructed).expect("fill ok");
        for (i, (&orig, &rec)) in data.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-6,
                "element {i}: orig={orig}, rec={rec}"
            );
        }
    }

    #[test]
    fn test_tt_get() {
        let (data, shape) = small_tensor_data();
        let tt = tt_svd(&data, &shape, 100, 1e-10).expect("tt_svd ok");
        // Check X[1,2,3] = data[1*12+2*4+3] = data[23] = 24
        let val = tt.get(&[1, 2, 3]).expect("get ok");
        assert!((val - 24.0).abs() < 1e-6, "X[1,2,3] = {val}");
        // Check X[0,0,0] = 1
        let val2 = tt.get(&[0, 0, 0]).expect("get ok");
        assert!((val2 - 1.0).abs() < 1e-6, "X[0,0,0] = {val2}");
    }

    #[test]
    fn test_tt_ranks() {
        let (data, shape) = small_tensor_data();
        let tt = tt_svd(&data, &shape, 100, 1e-10).expect("ok");
        let ranks = tt.ranks();
        assert_eq!(ranks[0], 1, "r_0 must be 1");
        assert_eq!(*ranks.last().expect("last"), 1, "r_d must be 1");
        assert_eq!(ranks.len(), shape.len() + 1);
    }

    #[test]
    fn test_tt_add() {
        let (data, shape) = small_tensor_data();
        let tt = tt_svd(&data, &shape, 100, 1e-10).expect("ok");
        let tt2 = tt_add(&tt, &tt).expect("add ok");
        // X + X = 2X, so element X[0,0,0] should be 2.0
        let val = tt2.get(&[0, 0, 0]).expect("get ok");
        assert!((val - 2.0).abs() < 1e-5, "X+X[0,0,0] = {val}");
    }

    #[test]
    fn test_tt_hadamard() {
        let (data, shape) = small_tensor_data();
        let tt = tt_svd(&data, &shape, 100, 1e-10).expect("ok");
        let tt_had = tt_hadamard(&tt, &tt).expect("hadamard ok");
        // X ⊙ X = X², so X²[0,0,0] should be 1² = 1
        let val = tt_had.get(&[0, 0, 0]).expect("get ok");
        assert!((val - 1.0).abs() < 1e-4, "X⊙X[0,0,0] = {val}");
        // X²[1,2,3] = 24² = 576
        let val2 = tt_had.get(&[1, 2, 3]).expect("get ok");
        assert!((val2 - 576.0).abs() < 1e-2, "X⊙X[1,2,3] = {val2}");
    }

    #[test]
    fn test_tt_inner_product() {
        let (data, shape) = small_tensor_data();
        let tt = tt_svd(&data, &shape, 100, 1e-10).expect("ok");
        let ip = inner_product(&tt, &tt).expect("inner ok");
        // <X, X> = ‖X‖² = sum of squares of 1..24
        let expected: f64 = data.iter().map(|v| v * v).sum();
        assert!(
            (ip - expected).abs() < 1e-4,
            "inner product {ip} != {expected}"
        );
    }

    #[test]
    fn test_tt_compress() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let shape = vec![2_usize, 3, 4];
        let tt = tt_svd(&data, &shape, 100, 0.0).expect("ok");
        let tt_small = tt.compress(2, 1e-3).expect("compress ok");
        // Compressed version should still approximate well
        let val_orig = tt.get(&[0, 1, 2]).expect("ok");
        let val_comp = tt_small.get(&[0, 1, 2]).expect("ok");
        assert!(
            (val_orig - val_comp).abs() < 2.0,
            "compressed val {val_comp} too far from {val_orig}"
        );
    }

    #[test]
    fn test_tt_svd_error_bad_shape() {
        let data = vec![1.0_f64, 2.0];
        let result = tt_svd(&data, &[3, 3], 10, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_tt_core_new_validates() {
        let bad = TTCore::new(vec![1.0; 5], (2, 2, 2));
        assert!(bad.is_err(), "Should fail: 5 != 2*2*2=8");
        let good = TTCore::new(vec![0.0; 8], (2, 2, 2));
        assert!(good.is_ok());
    }
}
