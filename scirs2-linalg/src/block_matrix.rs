//! Block matrix operations
//!
//! This module provides efficient block matrix storage and operations:
//!
//! - `BlockMatrix<T>`: Storage for m×n block partitioning of a matrix
//! - `block_matmul`: Cache-friendly block-wise matrix multiplication
//! - `block_triangular_solve`: Block back/forward substitution
//! - `schur_complement`: Compute S = D - C A⁻¹ B
//! - `block_cholesky`: Block Cholesky decomposition for structured SPD matrices
//! - `arrow_matrix_solve`: Arrow-structured linear system solver
//! - `bordered_system_solve`: Bordered block system via Schur complement

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use std::fmt::{Debug, Display};
use std::iter::Sum;

// ─────────────────────────────────────────────────────────────────────────────
// Trait alias
// ─────────────────────────────────────────────────────────────────────────────

/// Floating-point trait bounds required by every function in this module.
pub trait BlockFloat:
    Float + NumAssign + Debug + Display + ScalarOperand + Sum + 'static + Send + Sync
{
}

impl<T> BlockFloat for T where
    T: Float + NumAssign + Debug + Display + ScalarOperand + Sum + 'static + Send + Sync
{
}

// ─────────────────────────────────────────────────────────────────────────────
// BlockMatrix
// ─────────────────────────────────────────────────────────────────────────────

/// A matrix stored as an m×n arrangement of rectangular blocks.
///
/// The (i, j)-th block occupies rows `row_offsets[i]..row_offsets[i+1]` and
/// columns `col_offsets[j]..col_offsets[j+1]` of the full matrix.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::block_matrix::BlockMatrix;
///
/// let a = array![[1.0_f64, 2.0, 3.0],
///                [4.0, 5.0, 6.0],
///                [7.0, 8.0, 9.0]];
/// let bm = BlockMatrix::from_matrix_uniform(&a.view(), 2, 2).expect("ok");
/// assert_eq!(bm.block_rows(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct BlockMatrix<T> {
    /// Flattened storage of blocks in row-major block order.
    blocks: Vec<Array2<T>>,
    /// Number of block rows.
    block_rows: usize,
    /// Number of block columns.
    block_cols: usize,
    /// Row offset for each block row (length = block_rows + 1).
    row_offsets: Vec<usize>,
    /// Column offset for each block column (length = block_cols + 1).
    col_offsets: Vec<usize>,
}

impl<T: BlockFloat> BlockMatrix<T> {
    /// Create a `BlockMatrix` from a dense matrix using uniform block sizes.
    ///
    /// If `block_size_rows` or `block_size_cols` does not divide the matrix
    /// dimensions evenly, the last block row / column will be smaller.
    pub fn from_matrix_uniform(
        a: &ArrayView2<T>,
        block_size_rows: usize,
        block_size_cols: usize,
    ) -> LinalgResult<Self> {
        let total_rows = a.nrows();
        let total_cols = a.ncols();

        if block_size_rows == 0 || block_size_cols == 0 {
            return Err(LinalgError::ValueError(
                "block_size must be positive".into(),
            ));
        }

        let n_br = total_rows.div_ceil(block_size_rows);
        let n_bc = total_cols.div_ceil(block_size_cols);

        let row_offsets: Vec<usize> = (0..=n_br)
            .map(|i| (i * block_size_rows).min(total_rows))
            .collect();
        let col_offsets: Vec<usize> = (0..=n_bc)
            .map(|j| (j * block_size_cols).min(total_cols))
            .collect();

        Self::from_matrix_with_offsets(a, row_offsets, col_offsets)
    }

    /// Create a `BlockMatrix` from a dense matrix with explicit offset vectors.
    ///
    /// `row_offsets` must start at 0, end at `a.nrows()` and be strictly
    /// increasing. Same contract for `col_offsets`.
    pub fn from_matrix_with_offsets(
        a: &ArrayView2<T>,
        row_offsets: Vec<usize>,
        col_offsets: Vec<usize>,
    ) -> LinalgResult<Self> {
        let n_br = row_offsets.len().saturating_sub(1);
        let n_bc = col_offsets.len().saturating_sub(1);

        if row_offsets.is_empty() || col_offsets.is_empty() {
            return Err(LinalgError::ValueError(
                "offset vectors must not be empty".into(),
            ));
        }
        if *row_offsets
            .last()
            .ok_or_else(|| LinalgError::ValueError("row_offsets is empty".into()))?
            != a.nrows()
        {
            return Err(LinalgError::DimensionError(
                "row_offsets last element must equal a.nrows()".into(),
            ));
        }
        if *col_offsets
            .last()
            .ok_or_else(|| LinalgError::ValueError("col_offsets is empty".into()))?
            != a.ncols()
        {
            return Err(LinalgError::DimensionError(
                "col_offsets last element must equal a.ncols()".into(),
            ));
        }

        let mut blocks = Vec::with_capacity(n_br * n_bc);
        for bi in 0..n_br {
            for bj in 0..n_bc {
                let r0 = row_offsets[bi];
                let r1 = row_offsets[bi + 1];
                let c0 = col_offsets[bj];
                let c1 = col_offsets[bj + 1];
                blocks.push(a.slice(s![r0..r1, c0..c1]).to_owned());
            }
        }

        Ok(Self {
            blocks,
            block_rows: n_br,
            block_cols: n_bc,
            row_offsets,
            col_offsets,
        })
    }

    /// Return the number of block rows.
    #[inline]
    pub fn block_rows(&self) -> usize {
        self.block_rows
    }

    /// Return the number of block columns.
    #[inline]
    pub fn block_cols(&self) -> usize {
        self.block_cols
    }

    /// Access the (i, j)-th block as an immutable view.
    pub fn block(&self, i: usize, j: usize) -> LinalgResult<&Array2<T>> {
        if i >= self.block_rows || j >= self.block_cols {
            return Err(LinalgError::IndexError(format!(
                "block ({i},{j}) out of range ({},{} blocks)",
                self.block_rows, self.block_cols
            )));
        }
        Ok(&self.blocks[i * self.block_cols + j])
    }

    /// Access the (i, j)-th block mutably.
    pub fn block_mut(&mut self, i: usize, j: usize) -> LinalgResult<&mut Array2<T>> {
        if i >= self.block_rows || j >= self.block_cols {
            return Err(LinalgError::IndexError(format!(
                "block ({i},{j}) out of range"
            )));
        }
        Ok(&mut self.blocks[i * self.block_cols + j])
    }

    /// Convert back to a dense `Array2<T>`.
    pub fn to_dense(&self) -> Array2<T> {
        let total_rows = *self.row_offsets.last().unwrap_or(&0);
        let total_cols = *self.col_offsets.last().unwrap_or(&0);
        let mut out = Array2::zeros((total_rows, total_cols));
        for bi in 0..self.block_rows {
            for bj in 0..self.block_cols {
                let r0 = self.row_offsets[bi];
                let r1 = self.row_offsets[bi + 1];
                let c0 = self.col_offsets[bj];
                let c1 = self.col_offsets[bj + 1];
                let blk = &self.blocks[bi * self.block_cols + bj];
                out.slice_mut(s![r0..r1, c0..c1]).assign(blk);
            }
        }
        out
    }

    /// Total row count.
    #[inline]
    pub fn nrows(&self) -> usize {
        *self.row_offsets.last().unwrap_or(&0)
    }

    /// Total column count.
    #[inline]
    pub fn ncols(&self) -> usize {
        *self.col_offsets.last().unwrap_or(&0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// block_matmul
// ─────────────────────────────────────────────────────────────────────────────

/// Cache-friendly block matrix multiplication: C = A × B.
///
/// Splits the matrices into blocks of size `block_size` × `block_size` and
/// accumulates the partial products, keeping cache pressure low.
///
/// # Arguments
/// * `a`          - Left operand (m × k)
/// * `b`          - Right operand (k × n)
/// * `block_size` - Tile size in each dimension
///
/// # Returns
/// Product matrix C (m × n)
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::block_matrix::block_matmul;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
/// let c = block_matmul(&a.view(), &b.view(), 2).expect("ok");
/// assert!((c[[0,0]] - 19.0).abs() < 1e-10);
/// ```
pub fn block_matmul<T: BlockFloat>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
    block_size: usize,
) -> LinalgResult<Array2<T>> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    if k != b.nrows() {
        return Err(LinalgError::DimensionError(format!(
            "block_matmul: A is {m}×{k} but B is {}×{n}",
            b.nrows()
        )));
    }
    if block_size == 0 {
        return Err(LinalgError::ValueError("block_size must be > 0".into()));
    }

    let bs = block_size;
    let mut c = Array2::<T>::zeros((m, n));

    let mut i = 0;
    while i < m {
        let i_end = (i + bs).min(m);
        let mut j = 0;
        while j < n {
            let j_end = (j + bs).min(n);
            let mut p = 0;
            while p < k {
                let p_end = (p + bs).min(k);
                // c[i..i_end, j..j_end] += a[i..i_end, p..p_end] * b[p..p_end, j..j_end]
                let a_blk = a.slice(s![i..i_end, p..p_end]);
                let b_blk = b.slice(s![p..p_end, j..j_end]);
                let partial: Array2<T> = a_blk.dot(&b_blk);
                c.slice_mut(s![i..i_end, j..j_end])
                    .zip_mut_with(&partial, |c_el, &p_el| *c_el += p_el);
                p += bs;
            }
            j += bs;
        }
        i += bs;
    }

    Ok(c)
}

// ─────────────────────────────────────────────────────────────────────────────
// block_triangular_solve
// ─────────────────────────────────────────────────────────────────────────────

/// Solve a block triangular system L X = B (lower) or U X = B (upper).
///
/// Each diagonal block is inverted via Gaussian elimination; off-diagonal
/// contributions are subtracted from the right-hand side.
///
/// # Arguments
/// * `t`       - Block-triangular matrix (must be square with the same block
///   partition on rows and columns)
/// * `b`       - Right-hand side matrix (same total rows as `t`)
/// * `lower`   - `true` for lower triangular, `false` for upper triangular
/// * `offsets` - Block boundary offsets (e.g. `[0, k1, k2, n]`); must start
///   at 0 and end at `t.nrows()`
///
/// # Returns
/// Solution matrix X
pub fn block_triangular_solve<T: BlockFloat>(
    t: &ArrayView2<T>,
    b: &ArrayView2<T>,
    lower: bool,
    offsets: &[usize],
) -> LinalgResult<Array2<T>> {
    let n = t.nrows();
    if t.ncols() != n {
        return Err(LinalgError::ShapeError(
            "block_triangular_solve: T must be square".into(),
        ));
    }
    if b.nrows() != n {
        return Err(LinalgError::DimensionError(format!(
            "block_triangular_solve: T is {n}×{n} but B has {} rows",
            b.nrows()
        )));
    }
    if offsets.len() < 2 {
        return Err(LinalgError::ValueError(
            "offsets must have at least 2 elements".into(),
        ));
    }
    if offsets[0] != 0
        || *offsets
            .last()
            .ok_or_else(|| LinalgError::ValueError("offsets empty".into()))?
            != n
    {
        return Err(LinalgError::ValueError(
            "offsets must start at 0 and end at n".into(),
        ));
    }

    let nb = offsets.len() - 1; // number of block rows
    let rhs_cols = b.ncols();
    let mut x = b.to_owned();

    if lower {
        // Forward substitution: for i = 0..nb
        for i in 0..nb {
            let r0 = offsets[i];
            let r1 = offsets[i + 1];
            // Subtract contributions of already-solved blocks
            for j in 0..i {
                let c0 = offsets[j];
                let c1 = offsets[j + 1];
                let t_ij = t.slice(s![r0..r1, c0..c1]).to_owned();
                let x_j = x.slice(s![c0..c1, 0..rhs_cols]).to_owned();
                let contrib: Array2<T> = t_ij.dot(&x_j);
                x.slice_mut(s![r0..r1, 0..rhs_cols])
                    .zip_mut_with(&contrib, |v, &c| *v -= c);
            }
            // Solve diagonal block: T_ii * x_i = rhs_i
            let diag_blk = t.slice(s![r0..r1, r0..r1]).to_owned();
            let rhs_i = x.slice(s![r0..r1, 0..rhs_cols]).to_owned();
            let sol = solve_small_system(&diag_blk.view(), &rhs_i.view())?;
            x.slice_mut(s![r0..r1, 0..rhs_cols]).assign(&sol);
        }
    } else {
        // Backward substitution: for i = nb-1..=0
        let mut i = nb;
        loop {
            if i == 0 {
                break;
            }
            i -= 1;
            let r0 = offsets[i];
            let r1 = offsets[i + 1];
            for j in (i + 1)..nb {
                let c0 = offsets[j];
                let c1 = offsets[j + 1];
                let t_ij = t.slice(s![r0..r1, c0..c1]).to_owned();
                let x_j = x.slice(s![c0..c1, 0..rhs_cols]).to_owned();
                let contrib: Array2<T> = t_ij.dot(&x_j);
                x.slice_mut(s![r0..r1, 0..rhs_cols])
                    .zip_mut_with(&contrib, |v, &c| *v -= c);
            }
            let diag_blk = t.slice(s![r0..r1, r0..r1]).to_owned();
            let rhs_i = x.slice(s![r0..r1, 0..rhs_cols]).to_owned();
            let sol = solve_small_system(&diag_blk.view(), &rhs_i.view())?;
            x.slice_mut(s![r0..r1, 0..rhs_cols]).assign(&sol);
        }
    }

    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// schur_complement
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Schur complement S = D - C A⁻¹ B.
///
/// Given the block matrix
///
/// ```text
/// [ A  B ]
/// [ C  D ]
/// ```
///
/// the Schur complement of A is S = D - C A⁻¹ B.
///
/// # Arguments
/// * `a` - Upper-left block (p × p, must be invertible)
/// * `b` - Upper-right block (p × q)
/// * `c` - Lower-left block (q × p)
/// * `d` - Lower-right block (q × q)
///
/// # Returns
/// Schur complement S (q × q)
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::block_matrix::schur_complement;
///
/// let a = array![[2.0_f64, 0.0], [0.0, 2.0]];
/// let b = array![[1.0_f64], [1.0]];
/// let c = array![[1.0_f64, 1.0]];
/// let d = array![[3.0_f64]];
/// let s = schur_complement(&a.view(), &b.view(), &c.view(), &d.view()).expect("ok");
/// assert!((s[[0,0]] - 2.0).abs() < 1e-10); // 3 - 0.5 - 0.5 = 2
/// ```
pub fn schur_complement<T: BlockFloat>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
    c: &ArrayView2<T>,
    d: &ArrayView2<T>,
) -> LinalgResult<Array2<T>> {
    let p = a.nrows();
    if a.ncols() != p {
        return Err(LinalgError::ShapeError("A must be square".into()));
    }
    if b.nrows() != p {
        return Err(LinalgError::DimensionError(format!(
            "B must have {p} rows, got {}",
            b.nrows()
        )));
    }
    let q = c.nrows();
    if c.ncols() != p {
        return Err(LinalgError::DimensionError(format!(
            "C must have {p} cols, got {}",
            c.ncols()
        )));
    }
    if d.nrows() != q || d.ncols() != b.ncols() {
        return Err(LinalgError::DimensionError(format!(
            "D must be {q}×{}, got {}×{}",
            b.ncols(),
            d.nrows(),
            d.ncols()
        )));
    }

    // Solve A X = B  (i.e. X = A^{-1} B)
    let a_inv_b = solve_small_system(a, b)?;
    // S = D - C * (A^{-1} B)
    let c_a_inv_b: Array2<T> = c.dot(&a_inv_b);
    Ok(d.to_owned() - c_a_inv_b)
}

// ─────────────────────────────────────────────────────────────────────────────
// block_cholesky
// ─────────────────────────────────────────────────────────────────────────────

/// Block Cholesky decomposition for a symmetric positive-definite matrix.
///
/// Partitions the matrix into 2×2 blocks and applies the standard Cholesky
/// recursion at the block level.  The result is a block lower-triangular
/// factor `L` such that `L Lᵀ = A`.
///
/// # Arguments
/// * `a`          - Symmetric positive-definite matrix (n × n)
/// * `block_size` - Desired block size
///
/// # Returns
/// Lower triangular Cholesky factor L
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::block_matrix::block_cholesky;
///
/// let a = array![[4.0_f64, 2.0, 0.0],
///                [2.0, 5.0, 1.0],
///                [0.0, 1.0, 3.0]];
/// let l = block_cholesky(&a.view(), 2).expect("ok");
/// let lt = l.t().to_owned();
/// let reconstructed = l.dot(&lt);
/// for i in 0..3 { for j in 0..3 {
///     assert!((reconstructed[[i,j]] - a[[i,j]]).abs() < 1e-9);
/// }}
/// ```
pub fn block_cholesky<T: BlockFloat>(
    a: &ArrayView2<T>,
    block_size: usize,
) -> LinalgResult<Array2<T>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "block_cholesky: A must be square".into(),
        ));
    }
    if block_size == 0 {
        return Err(LinalgError::ValueError("block_size must be > 0".into()));
    }

    let mut l = Array2::<T>::zeros((n, n));

    // Build block offsets
    let offsets: Vec<usize> = {
        let mut v = vec![0usize];
        let mut cur = 0;
        while cur < n {
            cur = (cur + block_size).min(n);
            v.push(cur);
        }
        v
    };
    let nb = offsets.len() - 1;

    for i in 0..nb {
        let r0 = offsets[i];
        let r1 = offsets[i + 1];

        // Compute A_ii - sum_{k<i} L_ik L_ik^T
        let mut s_ii = a.slice(s![r0..r1, r0..r1]).to_owned();
        for k in 0..i {
            let c0 = offsets[k];
            let c1 = offsets[k + 1];
            let l_ik = l.slice(s![r0..r1, c0..c1]).to_owned();
            let contrib: Array2<T> = l_ik.dot(&l_ik.t());
            s_ii = s_ii - contrib;
        }

        // Cholesky of the diagonal block
        let l_ii = cholesky_small(&s_ii.view())?;
        l.slice_mut(s![r0..r1, r0..r1]).assign(&l_ii);

        // Off-diagonal blocks: L_ji = (A_ji - sum_{k<i} L_jk L_ik^T) L_ii^{-T}
        for j in (i + 1)..nb {
            let rj0 = offsets[j];
            let rj1 = offsets[j + 1];

            let mut s_ji = a.slice(s![rj0..rj1, r0..r1]).to_owned();
            for k in 0..i {
                let c0 = offsets[k];
                let c1 = offsets[k + 1];
                let l_jk = l.slice(s![rj0..rj1, c0..c1]).to_owned();
                let l_ik = l.slice(s![r0..r1, c0..c1]).to_owned();
                let contrib: Array2<T> = l_jk.dot(&l_ik.t());
                s_ji = s_ji - contrib;
            }

            // Solve L_ii^T X^T = S_ji^T  =>  X = S_ji L_ii^{-T}
            // Equivalently: X L_ii^T = S_ji  =>  L_ii X^T = S_ji^T
            let l_ii_view = l.slice(s![r0..r1, r0..r1]).to_owned();
            let l_ji = solve_lower_triangular_right(&l_ii_view.view(), &s_ji.view())?;
            l.slice_mut(s![rj0..rj1, r0..r1]).assign(&l_ji);
        }
    }

    Ok(l)
}

// ─────────────────────────────────────────────────────────────────────────────
// arrow_matrix_solve
// ─────────────────────────────────────────────────────────────────────────────

/// Solve a linear system with arrow (bordered diagonal) structure.
///
/// The arrow matrix has the form:
///
/// ```text
/// [ D   B ]   x = [ f ]
/// [ Bᵀ  C ]       [ g ]
/// ```
///
/// where D is diagonal (or block-diagonal, given as a vector of diagonal
/// values), B is (n × m), and C is (m × m).
///
/// # Arguments
/// * `d_diag` - Diagonal entries of D (n values)
/// * `b`      - Off-diagonal block B (n × m)
/// * `c`      - Corner block C (m × m)
/// * `f`      - Right-hand side for the top part (length n)
/// * `g`      - Right-hand side for the bottom part (length m)
///
/// # Returns
/// Solution vectors (x_top, x_bot) of lengths n and m respectively.
///
/// # Example
/// ```rust
/// use scirs2_linalg::block_matrix::arrow_matrix_solve;
///
/// // 3×3 arrow system: diag [2,2,2], B=[[1],[1],[1]], C=[[3]], f=[1,1,1], g=[1]
/// let d = vec![2.0_f64, 2.0, 2.0];
/// let b = scirs2_core::ndarray::array![[1.0_f64], [1.0], [1.0]];
/// let c = scirs2_core::ndarray::array![[3.0_f64]];
/// let f = vec![1.0_f64, 1.0, 1.0];
/// let g = vec![1.0_f64];
/// let (x, y) = arrow_matrix_solve(&d, &b.view(), &c.view(), &f, &g).expect("ok");
/// // Verify Dx + By = f and B^T x + Cy = g
/// for i in 0..3 { assert!((d[i]*x[i] + b[[i,0]]*y[0] - f[i]).abs() < 1e-9); }
/// ```
pub fn arrow_matrix_solve<T: BlockFloat>(
    d_diag: &[T],
    b: &ArrayView2<T>,
    c: &ArrayView2<T>,
    f: &[T],
    g: &[T],
) -> LinalgResult<(Vec<T>, Vec<T>)> {
    let n = d_diag.len();
    let m = c.nrows();

    if b.nrows() != n || b.ncols() != m {
        return Err(LinalgError::DimensionError(format!(
            "B must be {n}×{m}, got {}×{}",
            b.nrows(),
            b.ncols()
        )));
    }
    if c.ncols() != m {
        return Err(LinalgError::ShapeError("C must be square".into()));
    }
    if f.len() != n {
        return Err(LinalgError::DimensionError(format!(
            "f must have length {n}, got {}",
            f.len()
        )));
    }
    if g.len() != m {
        return Err(LinalgError::DimensionError(format!(
            "g must have length {m}, got {}",
            g.len()
        )));
    }

    // Check diagonal is non-zero
    for (i, &d) in d_diag.iter().enumerate() {
        if d.abs() < T::epsilon() {
            return Err(LinalgError::SingularMatrixError(format!(
                "D diagonal entry {i} is (near-)zero: {d:?}"
            )));
        }
    }

    // Strategy: block elimination via Schur complement of D.
    //   x = D^{-1}(f - B y)
    //   (C - B^T D^{-1} B) y = g - B^T D^{-1} f
    //
    // 1. Compute D^{-1} f and D^{-1} B
    let d_inv_f: Vec<T> = (0..n).map(|i| f[i] / d_diag[i]).collect();
    let mut d_inv_b = b.to_owned();
    for i in 0..n {
        for j in 0..m {
            d_inv_b[[i, j]] = b[[i, j]] / d_diag[i];
        }
    }

    // 2. Schur complement S = C - B^T D^{-1} B
    let bt_d_inv_b: Array2<T> = b.t().dot(&d_inv_b);
    let s: Array2<T> = c.to_owned() - bt_d_inv_b;

    // 3. RHS for y: g - B^T D^{-1} f
    let bt_d_inv_f: Vec<T> = (0..m)
        .map(|j| (0..n).map(|i| b[[i, j]] * d_inv_f[i]).sum())
        .collect();
    let rhs_y: Vec<T> = (0..m).map(|j| g[j] - bt_d_inv_f[j]).collect();

    // 4. Solve S y = rhs_y
    use scirs2_core::ndarray::Array1;
    let rhs_y_arr = Array1::from_vec(rhs_y);
    let y_arr = crate::solve::solve(&s.view(), &rhs_y_arr.view(), None)?;
    let y: Vec<T> = y_arr.iter().cloned().collect();

    // 5. Back-substitute: x = D^{-1}(f - B y)
    let x: Vec<T> = (0..n)
        .map(|i| {
            let by_i: T = (0..m).map(|j| b[[i, j]] * y[j]).sum();
            (f[i] - by_i) / d_diag[i]
        })
        .collect();

    Ok((x, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// bordered_system_solve
// ─────────────────────────────────────────────────────────────────────────────

/// Solve a bordered block system via Schur complement.
///
/// The bordered system has the form:
///
/// ```text
/// [ A  B ] [ x ]   [ f ]
/// [ C  D ] [ y ] = [ g ]
/// ```
///
/// where A is the (potentially large) principal block that is invertible.
/// Elimination proceeds through the Schur complement of A.
///
/// # Arguments
/// * `a` - Principal block (p × p)
/// * `b` - Upper-right block (p × q)
/// * `c` - Lower-left block (q × p)
/// * `d` - Lower-right block (q × q)
/// * `f` - Right-hand side for top rows (p × r)
/// * `g` - Right-hand side for bottom rows (q × r)
///
/// # Returns
/// Solution matrices (x, y) of shapes (p × r) and (q × r) respectively.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::block_matrix::bordered_system_solve;
///
/// let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
/// let b = array![[1.0_f64], [1.0]];
/// let c = array![[1.0_f64, 1.0]];
/// let d = array![[5.0_f64]];
/// let f = array![[1.0_f64], [2.0]];
/// let g = array![[3.0_f64]];
/// let (x, y) = bordered_system_solve(&a.view(), &b.view(), &c.view(), &d.view(),
///                                     &f.view(), &g.view()).expect("ok");
/// // verify A x + B y = f
/// let resid = a.dot(&x) + b.dot(&y) - &f;
/// for v in resid.iter() { assert!(v.abs() < 1e-9); }
/// ```
pub fn bordered_system_solve<T: BlockFloat>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
    c: &ArrayView2<T>,
    d: &ArrayView2<T>,
    f: &ArrayView2<T>,
    g: &ArrayView2<T>,
) -> LinalgResult<(Array2<T>, Array2<T>)> {
    let p = a.nrows();
    let q = c.nrows();
    let r = f.ncols();

    if a.ncols() != p {
        return Err(LinalgError::ShapeError("A must be square".into()));
    }
    if b.nrows() != p || b.ncols() != q {
        return Err(LinalgError::DimensionError(format!(
            "B must be {p}×{q}, got {}×{}",
            b.nrows(),
            b.ncols()
        )));
    }
    if c.ncols() != p {
        return Err(LinalgError::DimensionError("C col count mismatch".into()));
    }
    if d.nrows() != q || d.ncols() != q {
        return Err(LinalgError::DimensionError("D must be q×q".into()));
    }
    if f.nrows() != p || f.ncols() != r {
        return Err(LinalgError::DimensionError("f must be p×r".into()));
    }
    if g.nrows() != q || g.ncols() != r {
        return Err(LinalgError::DimensionError("g must be q×r".into()));
    }

    // Solve A X_f = f  and  A X_b = B
    let a_inv_f = solve_small_system(a, f)?;
    let a_inv_b = solve_small_system(a, b)?;

    // Schur complement: S = D - C A^{-1} B
    let c_a_inv_b: Array2<T> = c.dot(&a_inv_b);
    let s: Array2<T> = d.to_owned() - c_a_inv_b;

    // Modified RHS for y: g - C A^{-1} f
    let c_a_inv_f: Array2<T> = c.dot(&a_inv_f);
    let rhs_y: Array2<T> = g.to_owned() - c_a_inv_f;

    // Solve S y = rhs_y
    let y = solve_small_system(&s.view(), &rhs_y.view())?;

    // Back-substitute: x = A^{-1}(f - B y)
    let b_y: Array2<T> = b.dot(&y);
    let rhs_x: Array2<T> = f.to_owned() - b_y;
    let x = solve_small_system(a, &rhs_x.view())?;

    Ok((x, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Solve a (small) linear system A X = B using the crate's standard solver.
/// B may be a matrix (multiple right-hand sides).
pub(crate) fn solve_small_system<T: BlockFloat>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
) -> LinalgResult<Array2<T>> {
    let n = a.nrows();
    let m = b.ncols();
    let mut result = Array2::<T>::zeros((n, m));
    for j in 0..m {
        let col = b.column(j).to_owned();
        let sol = crate::solve::solve(a, &col.view(), None)?;
        result.column_mut(j).assign(&sol);
    }
    Ok(result)
}

/// Cholesky decomposition for a small dense matrix (Banachiewicz algorithm).
fn cholesky_small<T: BlockFloat>(a: &ArrayView2<T>) -> LinalgResult<Array2<T>> {
    let n = a.nrows();
    let mut l = Array2::<T>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = T::zero();
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let diff = a[[i, i]] - sum;
                if diff <= T::zero() {
                    return Err(LinalgError::NonPositiveDefiniteError(format!(
                        "block_cholesky: diagonal element ({i},{i}) is non-positive during factorization"
                    )));
                }
                l[[i, j]] = diff.sqrt();
            } else {
                let l_jj = l[[j, j]];
                if l_jj.abs() < T::epsilon() {
                    return Err(LinalgError::SingularMatrixError(format!(
                        "cholesky_small: near-zero pivot at ({j},{j})"
                    )));
                }
                l[[i, j]] = (a[[i, j]] - sum) / l_jj;
            }
        }
    }
    Ok(l)
}

/// Solve X L^T = B for X, where L is lower triangular.
/// Equivalent to: for each row of X (and B), back-substitution with L^T.
fn solve_lower_triangular_right<T: BlockFloat>(
    l: &ArrayView2<T>,
    b: &ArrayView2<T>,
) -> LinalgResult<Array2<T>> {
    // X L^T = B  <=>  L X^T = B^T
    let bt = b.t().to_owned();
    let xt = solve_lower_triangular_system(l, &bt.view())?;
    Ok(xt.t().to_owned())
}

/// Forward substitution: solve L X = B where L is lower triangular.
fn solve_lower_triangular_system<T: BlockFloat>(
    l: &ArrayView2<T>,
    b: &ArrayView2<T>,
) -> LinalgResult<Array2<T>> {
    let n = l.nrows();
    let m = b.ncols();
    let mut x = Array2::<T>::zeros((n, m));
    for j in 0..m {
        for i in 0..n {
            let mut val = b[[i, j]];
            for k in 0..i {
                val -= l[[i, k]] * x[[k, j]];
            }
            let l_ii = l[[i, i]];
            if l_ii.abs() < T::epsilon() {
                return Err(LinalgError::SingularMatrixError(format!(
                    "solve_lower_triangular: zero pivot at ({i},{i})"
                )));
            }
            x[[i, j]] = val / l_ii;
        }
    }
    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_block_matrix_roundtrip() {
        let a = array![
            [1.0_f64, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];
        let bm = BlockMatrix::from_matrix_uniform(&a.view(), 2, 2).expect("ok");
        let dense = bm.to_dense();
        assert_abs_diff_eq!(a, dense, epsilon = 1e-12);
        assert_eq!(bm.block_rows(), 2);
        assert_eq!(bm.block_cols(), 2);
    }

    #[test]
    fn test_block_matmul() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
        let c_ref = a.dot(&b);
        let c_blk = block_matmul(&a.view(), &b.view(), 1).expect("ok");
        assert_abs_diff_eq!(c_ref, c_blk, epsilon = 1e-12);
        let c_blk2 = block_matmul(&a.view(), &b.view(), 2).expect("ok");
        assert_abs_diff_eq!(c_ref, c_blk2, epsilon = 1e-12);
    }

    #[test]
    fn test_schur_complement() {
        let a = array![[2.0_f64, 0.0], [0.0, 2.0]];
        let b = array![[1.0_f64], [1.0]];
        let c = array![[1.0_f64, 1.0]];
        let d = array![[3.0_f64]];
        let s = schur_complement(&a.view(), &b.view(), &c.view(), &d.view()).expect("ok");
        // S = 3 - [1,1] [0.5;0.5] = 3 - 1 = 2
        assert_abs_diff_eq!(s[[0, 0]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_block_cholesky() {
        let a = array![[4.0_f64, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 3.0]];
        let l = block_cholesky(&a.view(), 2).expect("ok");
        let lt = l.t().to_owned();
        let reconstructed = l.dot(&lt);
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_bordered_system_solve() {
        let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
        let b = array![[1.0_f64], [1.0]];
        let c = array![[1.0_f64, 1.0]];
        let d = array![[5.0_f64]];
        let f = array![[1.0_f64], [2.0]];
        let g = array![[3.0_f64]];
        let (x, y) = bordered_system_solve(
            &a.view(),
            &b.view(),
            &c.view(),
            &d.view(),
            &f.view(),
            &g.view(),
        )
        .expect("ok");
        let resid_top = a.dot(&x) + b.dot(&y) - &f;
        let resid_bot = c.dot(&x) + d.dot(&y) - &g;
        for v in resid_top.iter().chain(resid_bot.iter()) {
            assert!(v.abs() < 1e-9, "residual {v} too large");
        }
    }

    #[test]
    fn test_arrow_matrix_solve() {
        let d = vec![2.0_f64, 2.0, 2.0];
        let b = array![[1.0_f64], [1.0], [1.0]];
        let c = array![[3.0_f64]];
        let f = vec![1.0_f64, 1.0, 1.0];
        let g = vec![1.0_f64];
        let (x, y) = arrow_matrix_solve(&d, &b.view(), &c.view(), &f, &g).expect("ok");
        for i in 0..3 {
            let resid = d[i] * x[i] + b[[i, 0]] * y[0] - f[i];
            assert!(resid.abs() < 1e-9, "top resid {resid}");
        }
        let bot_resid: f64 =
            (0..3).map(|i| b[[i, 0]] * x[i]).sum::<f64>() + c[[0, 0]] * y[0] - g[0];
        assert!(bot_resid.abs() < 1e-9, "bot resid {bot_resid}");
    }

    #[test]
    fn test_block_triangular_solve_lower() {
        // L = [[2,0,0],[1,3,0],[0,2,4]]
        let t = array![[2.0_f64, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 2.0, 4.0]];
        let b = array![[2.0_f64], [5.0], [12.0]];
        let offsets = vec![0, 1, 2, 3];
        let x = block_triangular_solve(&t.view(), &b.view(), true, &offsets).expect("ok");
        let resid = t.dot(&x) - &b;
        for v in resid.iter() {
            assert!(v.abs() < 1e-9);
        }
    }

    #[test]
    fn test_block_triangular_solve_upper() {
        let t = array![[2.0_f64, 1.0, 0.0], [0.0, 3.0, 2.0], [0.0, 0.0, 4.0]];
        let b = array![[3.0_f64], [11.0], [8.0]];
        let offsets = vec![0, 1, 2, 3];
        let x = block_triangular_solve(&t.view(), &b.view(), false, &offsets).expect("ok");
        let resid = t.dot(&x) - &b;
        for v in resid.iter() {
            assert!(v.abs() < 1e-9);
        }
    }
}
