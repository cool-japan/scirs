//! Sparse Jacobian optimization
//!
//! This module provides functionality for detecting and exploiting sparsity
//! patterns in Jacobian matrices to improve computational efficiency.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::parallel_ops::*;
use std::collections::{HashMap, HashSet};

/// Represents a sparsity pattern
#[derive(Debug, Clone)]
pub struct SparsePattern {
    /// Non-zero entries as (row, col) pairs
    pub entries: Vec<(usize, usize)>,
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Row-wise non-zero indices
    pub row_indices: Vec<Vec<usize>>,
    /// Column-wise non-zero indices
    pub col_indices: Vec<Vec<usize>>,
}

impl SparsePattern {
    /// Create a new sparse pattern
    pub fn new(_nrows: usize, ncols: usize) -> Self {
        SparsePattern {
            entries: Vec::new(),
            nrows: _nrows,
            n_cols: ncols,
            row_indices: vec![Vec::new(); _nrows],
            col_indices: vec![Vec::new(); ncols],
        }
    }

    /// Add a non-zero entry
    pub fn add_entry(&mut self, row: usize, col: usize) {
        if row < self.nrows && col < self.n_cols {
            self.entries.push((row, col));
            self.row_indices[row].push(col);
            self.col_indices[col].push(row);
        }
    }

    /// Get the number of non-zeros
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Get the sparsity ratio (fraction of zeros)
    pub fn sparsity(&self) -> f64 {
        let total = (self.nrows * self.n_cols) as f64;
        if total > 0.0 {
            1.0 - (self.nnz() as f64 / total)
        } else {
            0.0
        }
    }

    /// Check if pattern is sparse enough to benefit from sparse methods
    pub fn is_sparse(&self, threshold: f64) -> bool {
        self.sparsity() > threshold
    }

    /// Compute coloring for efficient Jacobian computation
    pub fn compute_coloring(&self) -> ColGrouping {
        // Use improved Welsh-Powell algorithm for better coloring
        let mut degrees: Vec<(usize, usize)> = Vec::new();

        // Calculate degree of each column (number of structural neighbors)
        for col in 0..self.n_cols {
            let mut neighbors = HashSet::new();
            for &row in &self.col_indices[col] {
                for &other_col in &self.row_indices[row] {
                    if other_col != col {
                        neighbors.insert(other_col);
                    }
                }
            }
            degrees.push((col, neighbors.len()));
        }

        // Sort by degree (descending)
        degrees.sort_by_key(|&(_, deg)| std::cmp::Reverse(deg));

        let mut colors: HashMap<usize, usize> = HashMap::new();
        let mut max_color = 0;

        // Color vertices in order of decreasing degree
        for (col_, _) in degrees {
            let mut used_colors = HashSet::new();

            // Find colors used by adjacent columns
            for &row in &self.col_indices[col_] {
                for &other_col in &self.row_indices[row] {
                    if other_col != col_ {
                        if let Some(&color) = colors.get(&other_col) {
                            used_colors.insert(color);
                        }
                    }
                }
            }

            // Find first available color
            let mut color = 0;
            while used_colors.contains(&color) {
                color += 1;
            }

            colors.insert(col_, color);
            max_color = max_color.max(color);
        }

        // Group columns by color
        let mut groups = vec![Vec::new(); max_color + 1];
        for (&col, &color) in &colors {
            groups[color].push(col);
        }

        ColGrouping { groups }
    }
}

/// Column grouping for efficient Jacobian computation
pub struct ColGrouping {
    /// Groups of columns that can be computed together
    pub groups: Vec<Vec<usize>>,
}

impl ColGrouping {
    /// Get the number of groups
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }
}

/// Sparse Jacobian representation
pub struct SparseJacobian<F: IntegrateFloat> {
    /// The sparsity pattern
    pub pattern: SparsePattern,
    /// Non-zero values in row-major order
    pub values: Vec<F>,
    /// Mapping from (row, col) to value index
    pub index_map: HashMap<(usize, usize), usize>,
}

impl<F: IntegrateFloat> SparseJacobian<F> {
    /// Create a new sparse Jacobian
    pub fn new(pattern: SparsePattern) -> Self {
        let nnz = pattern.nnz();
        let mut index_map = HashMap::new();

        for (i, &(row, col)) in pattern.entries.iter().enumerate() {
            index_map.insert((row, col), i);
        }

        SparseJacobian {
            pattern,
            values: vec![F::zero(); nnz],
            index_map,
        }
    }

    /// Set a value in the sparse Jacobian
    pub fn set(&mut self, row: usize, col: usize, value: F) -> IntegrateResult<()> {
        if let Some(&idx) = self.index_map.get(&(row, col)) {
            self.values[idx] = value;
            Ok(())
        } else {
            Err(IntegrateError::IndexError(format!(
                "Entry ({row}, {col}) not in sparsity pattern"
            )))
        }
    }

    /// Get a value from the sparse Jacobian
    pub fn get(&self, row: usize, col: usize) -> Option<F> {
        self.index_map.get(&(row, col)).map(|&idx| self.values[idx])
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> Array2<F> {
        let mut dense = Array2::zeros((self.pattern.nrows, self.pattern.n_cols));
        for (&(row, col), &idx) in &self.index_map {
            dense[[row, col]] = self.values[idx];
        }
        dense
    }

    /// Multiply by a vector: y = J * x
    pub fn matvec(&self, x: ArrayView1<F>) -> IntegrateResult<Array1<F>> {
        if x.len() != self.pattern.n_cols {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} columns, got {}",
                self.pattern.n_cols,
                x.len()
            )));
        }

        let mut y = Array1::zeros(self.pattern.nrows);

        for (&(row, col), &idx) in &self.index_map {
            y[row] += self.values[idx] * x[col];
        }

        Ok(y)
    }

    /// Transpose multiply by a vector: y = J^T * x
    pub fn matvec_transpose(&self, x: ArrayView1<F>) -> IntegrateResult<Array1<F>> {
        if x.len() != self.pattern.nrows {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} rows, got {}",
                self.pattern.nrows,
                x.len()
            )));
        }

        let mut y = Array1::zeros(self.pattern.n_cols);

        for (&(row, col), &idx) in &self.index_map {
            y[col] += self.values[idx] * x[row];
        }

        Ok(y)
    }
}

/// Compressed Sparse Row (CSR) format for efficient operations
pub struct CSRJacobian<F: IntegrateFloat> {
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Row pointers (size nrows + 1)
    pub row_ptr: Vec<usize>,
    /// Column indices (size nnz)
    pub col_idx: Vec<usize>,
    /// Non-zero values (size nnz)
    pub values: Vec<F>,
}

/// Compressed Sparse Column (CSC) format for efficient column operations
pub struct CSCJacobian<F: IntegrateFloat> {
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Column pointers (size n_cols + 1)
    pub col_ptr: Vec<usize>,
    /// Row indices (size nnz)
    pub row_idx: Vec<usize>,
    /// Non-zero values (size nnz)
    pub values: Vec<F>,
}

impl<F: IntegrateFloat> SparseJacobian<F> {
    /// Create a new sparse Jacobian from pattern
    pub fn from_pattern(pattern: SparsePattern) -> Self {
        let mut index_map = HashMap::new();
        for (idx, &(row, col)) in pattern.entries.iter().enumerate() {
            index_map.insert((row, col), idx);
        }

        SparseJacobian {
            values: vec![F::zero(); pattern.entries.len()],
            pattern,
            index_map,
        }
    }

    /// Set a value without error checking
    pub fn set_unchecked(&mut self, row: usize, col: usize, value: F) {
        if let Some(&idx) = self.index_map.get(&(row, col)) {
            self.values[idx] = value;
        }
    }

    /// Get a value with default zero
    pub fn get_or_zero(&self, row: usize, col: usize) -> F {
        if let Some(&idx) = self.index_map.get(&(row, col)) {
            self.values[idx]
        } else {
            F::zero()
        }
    }

    /// Convert to dense matrix (alternative implementation)
    pub fn to_dense_alt(&self) -> Array2<F> {
        let mut dense = Array2::zeros((self.pattern.nrows, self.pattern.n_cols));
        for (idx, &(row, col)) in self.pattern.entries.iter().enumerate() {
            dense[[row, col]] = self.values[idx];
        }
        dense
    }

    /// Apply to vector (matrix-vector multiplication)
    pub fn apply(&self, x: ArrayView1<F>) -> IntegrateResult<Array1<F>> {
        if x.len() != self.pattern.n_cols {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} columns, got {}",
                self.pattern.n_cols,
                x.len()
            )));
        }

        let mut result = Array1::zeros(self.pattern.nrows);
        for (idx, &(row, col)) in self.pattern.entries.iter().enumerate() {
            result[row] += self.values[idx] * x[col];
        }

        Ok(result)
    }

    /// Convert to CSR format for efficient row operations
    pub fn to_csr(&self) -> CSRJacobian<F> {
        let mut entries: Vec<(usize, usize, F)> = Vec::new();
        for (idx, &(row, col)) in self.pattern.entries.iter().enumerate() {
            entries.push((row, col, self.values[idx]));
        }

        // Sort by row, then column
        entries.sort_by_key(|&(r, c, _)| (r, c));

        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        let mut currentrow = 0;
        for (row, col, val) in entries {
            while currentrow < row {
                row_ptr.push(col_idx.len());
                currentrow += 1;
            }
            col_idx.push(col);
            values.push(val);
        }

        // Fill remaining row pointers
        while row_ptr.len() <= self.pattern.nrows {
            row_ptr.push(col_idx.len());
        }

        CSRJacobian {
            nrows: self.pattern.nrows,
            n_cols: self.pattern.n_cols,
            row_ptr,
            col_idx,
            values,
        }
    }
}

/// Detect sparsity pattern by probing with finite differences
#[allow(dead_code)]
pub fn detect_sparsity<F, Func>(f: Func, x: ArrayView1<F>, eps: F) -> IntegrateResult<SparsePattern>
where
    F: IntegrateFloat + Send + Sync,
    Func: Fn(ArrayView1<F>) -> IntegrateResult<Array1<F>> + Sync,
{
    let n = x.len();
    let f0 = f(x)?;
    let m = f0.len();

    let mut pattern = SparsePattern::new(m, n);

    // Parallel probing with batched perturbations
    let results: Vec<_> = (0..n)
        .collect::<Vec<_>>()
        .par_chunks(n / scirs2_core::parallel_ops::num_threads().max(1) + 1)
        .map(|chunk| {
            let mut local_entries = Vec::new();
            for &j in chunk {
                let mut x_pert = x.to_owned();
                x_pert[j] += eps;
                if let Ok(f_pert) = f(x_pert.view()) {
                    for i in 0..m {
                        if (f_pert[i] - f0[i]).abs() > F::epsilon() {
                            local_entries.push((i, j));
                        }
                    }
                }
            }
            local_entries
        })
        .collect();

    // Merge results
    for entries in results {
        for (i, j) in entries {
            pattern.add_entry(i, j);
        }
    }

    Ok(pattern)
}

impl<F: IntegrateFloat + Send + Sync> CSRJacobian<F> {
    /// Apply to vector (matrix-vector multiplication) - optimized for CSR
    pub fn apply(&self, x: ArrayView1<F>) -> IntegrateResult<Array1<F>> {
        if x.len() != self.n_cols {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} columns, got {}",
                self.n_cols,
                x.len()
            )));
        }

        let mut result = Array1::zeros(self.nrows);

        // Parallel row-wise computation
        let chunk_size = (self.nrows / scirs2_core::parallel_ops::num_threads()).max(1);
        let chunks: Vec<_> = (0..self.nrows)
            .collect::<Vec<_>>()
            .par_chunks(chunk_size)
            .map(|rows| {
                let mut local_result = Array1::zeros(rows.len());
                for (local_idx, &row) in rows.iter().enumerate() {
                    let start = self.row_ptr[row];
                    let end = self.row_ptr[row + 1];
                    for idx in start..end {
                        local_result[local_idx] += self.values[idx] * x[self.col_idx[idx]];
                    }
                }
                (rows[0], local_result)
            })
            .collect();

        // Combine results
        for (startrow, chunk) in chunks {
            for (i, val) in chunk.iter().enumerate() {
                result[startrow + i] = *val;
            }
        }

        Ok(result)
    }

    /// Transpose to CSC format
    pub fn transpose(&self) -> CSCJacobian<F> {
        let mut entries: Vec<(usize, usize, F)> = Vec::new();

        for row in 0..self.nrows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for idx in start..end {
                entries.push((self.col_idx[idx], row, self.values[idx]));
            }
        }

        // Sort by column, then row
        entries.sort_by_key(|&(c, r, _)| (c, r));

        let mut col_ptr = vec![0];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();

        let mut current_col = 0;
        for (col, row, val) in entries {
            while current_col < col {
                col_ptr.push(row_idx.len());
                current_col += 1;
            }
            row_idx.push(row);
            values.push(val);
        }

        // Fill remaining column pointers
        while col_ptr.len() <= self.n_cols {
            col_ptr.push(row_idx.len());
        }

        CSCJacobian {
            nrows: self.n_cols,
            n_cols: self.nrows,
            col_ptr,
            row_idx,
            values,
        }
    }
}

/// Compress a dense Jacobian using a sparsity pattern
#[allow(dead_code)]
pub fn compress_jacobian<F: IntegrateFloat>(
    dense: ArrayView2<F>,
    pattern: &SparsePattern,
) -> SparseJacobian<F> {
    let mut sparse = SparseJacobian::from_pattern(pattern.clone());

    for (idx, &(row, col)) in pattern.entries.iter().enumerate() {
        sparse.values[idx] = dense[[row, col]];
    }

    sparse
}

/// Compute sparse Jacobian using coloring
#[allow(dead_code)]
pub fn colored_jacobian<F, Func>(
    f: Func,
    x: ArrayView1<F>,
    pattern: &SparsePattern,
    eps: F,
) -> IntegrateResult<SparseJacobian<F>>
where
    F: IntegrateFloat,
    Func: Fn(ArrayView1<F>) -> IntegrateResult<Array1<F>>,
{
    let coloring = pattern.compute_coloring();
    let f0 = f(x)?;
    let mut jacobian = SparseJacobian::from_pattern(pattern.clone());

    // Compute Jacobian using column groups
    for group in &coloring.groups {
        let mut x_pert = x.to_owned();

        // Perturb all columns in this group
        for &col in group {
            x_pert[col] += eps;
        }

        let f_pert = f(x_pert.view())?;

        // Extract derivatives for this group
        for &col in group {
            for &row in &pattern.col_indices[col] {
                let deriv = (f_pert[row] - f0[row]) / eps;
                let _ = jacobian.set(row, col, deriv);
            }
        }
    }

    Ok(jacobian)
}

/// Example: Create a tridiagonal sparsity pattern
#[allow(dead_code)]
pub fn example_tridiagonal_pattern(n: usize) -> SparsePattern {
    let mut pattern = SparsePattern::new(n, n);

    for i in 0..n {
        pattern.add_entry(i, i); // Diagonal
        if i > 0 {
            pattern.add_entry(i, i - 1); // Sub-diagonal
        }
        if i < n - 1 {
            pattern.add_entry(i, i + 1); // Super-diagonal
        }
    }

    pattern
}

/// Sparse Jacobian updater for quasi-Newton methods
pub struct SparseJacobianUpdater<F: IntegrateFloat> {
    pattern: SparsePattern,
    threshold: F,
}

impl<F: IntegrateFloat> SparseJacobianUpdater<F> {
    /// Create a new updater
    pub fn new(pattern: SparsePattern, threshold: F) -> Self {
        SparseJacobianUpdater { pattern, threshold }
    }

    /// Update sparse Jacobian using Broyden's method
    pub fn broyden_update(
        &self,
        jac: &mut SparseJacobian<F>,
        dx: ArrayView1<F>,
        df: ArrayView1<F>,
    ) -> IntegrateResult<()> {
        let jdx = jac.apply(dx)?;
        let dy = &df - &jdx;

        let dx_norm_sq = dx.dot(&dx);
        if dx_norm_sq < self.threshold {
            return Ok(());
        }

        // Update only non-zero entries
        for (idx, &(i, j)) in self.pattern.entries.iter().enumerate() {
            jac.values[idx] += dy[i] * dx[j] / dx_norm_sq;
        }

        Ok(())
    }
}

/// Adaptive sparsity detection with multiple perturbation sizes
#[allow(dead_code)]
pub fn detect_sparsity_adaptive<F, Func>(
    f: Func,
    x: ArrayView1<F>,
    eps_range: (F, F),
    n_samples: usize,
) -> IntegrateResult<SparsePattern>
where
    F: IntegrateFloat + Send + Sync,
    Func: Fn(ArrayView1<F>) -> IntegrateResult<Array1<F>> + Sync,
{
    let n = x.len();
    let f0 = f(x)?;
    let m = f0.len();

    let mut accumulated_pattern = SparsePattern::new(m, n);
    let eps_min = eps_range.0;
    let eps_max = eps_range.1;

    // Try multiple perturbation sizes
    for sample in 0..n_samples {
        let alpha = F::from(sample).unwrap() / F::from(n_samples - 1).unwrap();
        let eps = eps_min * (F::one() - alpha) + eps_max * alpha;

        let pattern = detect_sparsity(&f, x, eps)?;

        // Merge patterns
        for &(i, j) in &pattern.entries {
            accumulated_pattern.add_entry(i, j);
        }
    }

    Ok(accumulated_pattern)
}

/// Block-structured sparsity pattern
pub struct BlockPattern {
    /// Block sizes (rows, cols)
    pub block_sizes: Vec<(usize, usize)>,
    /// Non-zero blocks as (blockrow, block_col) pairs
    pub blocks: Vec<(usize, usize)>,
    /// Total rows and columns
    pub nrows: usize,
    pub n_cols: usize,
}

impl BlockPattern {
    /// Convert block pattern to regular sparsity pattern
    pub fn to_sparse_pattern(&self) -> SparsePattern {
        let mut pattern = SparsePattern::new(self.nrows, self.n_cols);

        let mut row_offset = 0;
        let mut col_offset = 0;

        for &(blockrow, block_col) in &self.blocks {
            let (block_height, block_width) = self.block_sizes[blockrow];

            // Add all entries in this block
            for i in 0..block_height {
                for j in 0..block_width {
                    pattern.add_entry(row_offset + i, col_offset + j);
                }
            }

            col_offset += block_width;
            if block_col == self.blocks.len() - 1 {
                col_offset = 0;
                row_offset += block_height;
            }
        }

        pattern
    }
}

/// Hybrid sparse format for mixed dense/sparse blocks
pub struct HybridJacobian<F: IntegrateFloat> {
    /// Sparse blocks in CSR format
    pub sparse_blocks: Vec<CSRJacobian<F>>,
    /// Dense blocks
    pub dense_blocks: Vec<Array2<F>>,
    /// Block layout information
    pub block_info: BlockPattern,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SparseJacobian, SparsePattern};
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_sparse_pattern() {
        let mut pattern = SparsePattern::new(3, 3);
        pattern.add_entry(0, 0);
        pattern.add_entry(1, 1);
        pattern.add_entry(2, 2);
        pattern.add_entry(0, 1);

        assert_eq!(pattern.nnz(), 4);
        assert!(pattern.sparsity() > 0.5);
    }

    #[test]
    fn test_coloring() {
        let pattern = example_tridiagonal_pattern(5);
        let coloring = pattern.compute_coloring();

        // Tridiagonal matrix should need at most 3 colors
        assert!(coloring.n_groups() <= 3);
    }

    #[test]
    fn test_sparse_jacobian() {
        let pattern = example_tridiagonal_pattern(3);
        let mut jac = SparseJacobian::from_pattern(pattern);

        // Set some values
        let _ = jac.set(0, 0, 2.0);
        let _ = jac.set(0, 1, -1.0);
        let _ = jac.set(1, 0, -1.0);
        let _ = jac.set(1, 1, 2.0);
        let _ = jac.set(1, 2, -1.0);
        let _ = jac.set(2, 1, -1.0);
        let _ = jac.set(2, 2, 2.0);

        // Test matrix-vector multiplication
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = jac.apply(x.view()).unwrap();

        // Should compute [2*1 - 1*2, -1*1 + 2*2 - 1*3, -1*2 + 2*3]
        assert!((y[0] - 0.0_f64).abs() < 1e-10);
        assert!((y[1] - 0.0_f64).abs() < 1e-10);
        assert!((y[2] - 4.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_csr_format() {
        let pattern = example_tridiagonal_pattern(3);
        let mut jac = SparseJacobian::from_pattern(pattern);

        // Set values
        let _ = jac.set(0, 0, 2.0);
        let _ = jac.set(0, 1, -1.0);
        let _ = jac.set(1, 0, -1.0);
        let _ = jac.set(1, 1, 2.0);
        let _ = jac.set(1, 2, -1.0);
        let _ = jac.set(2, 1, -1.0);
        let _ = jac.set(2, 2, 2.0);

        // Convert to CSR
        let csr = jac.to_csr();

        // Test CSR matrix-vector multiplication
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = csr.apply(x.view()).unwrap();

        assert!((y[0] - 0.0_f64).abs() < 1e-10);
        assert!((y[1] - 0.0_f64).abs() < 1e-10);
        assert!((y[2] - 4.0_f64).abs() < 1e-10);
    }
}
