//! Sparse matrix representations for memory-efficient text processing
//!
//! This module provides sparse matrix implementations optimized for text data
//! where most values are zero (common in TF-IDF and count vectorization).

use crate::error::{Result, TextError};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Compressed Sparse Row (CSR) matrix representation
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// Non-zero values
    values: Vec<f64>,
    /// Column indices for each value
    colindices: Vec<usize>,
    /// Row pointers (cumulative sum of non-zeros per row)
    row_ptrs: Vec<usize>,
    /// Number of rows
    n_rows: usize,
    /// Number of columns
    n_cols: usize,
}

impl CsrMatrix {
    /// Create a new CSR matrix from dense representation
    pub fn from_dense(dense: &Array2<f64>) -> Self {
        let (n_rows, n_cols) = dense.dim();
        let mut values = Vec::new();
        let mut colindices = Vec::new();
        let mut row_ptrs = vec![0];

        for row in dense.rows() {
            for (colidx, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    values.push(value);
                    colindices.push(colidx);
                }
            }
            row_ptrs.push(values.len());
        }

        Self {
            values,
            colindices,
            row_ptrs,
            n_rows,
            n_cols,
        }
    }

    /// Create an empty CSR matrix
    pub fn zeros(_n_rows: usize, ncols: usize) -> Self {
        Self {
            values: Vec::new(),
            colindices: Vec::new(),
            row_ptrs: vec![0; _n_rows + 1],
            n_rows: _n_rows,
            n_cols: ncols,
        }
    }

    /// Get the shape of the matrix
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        self.values.len() * std::mem::size_of::<f64>()
            + self.colindices.len() * std::mem::size_of::<usize>()
            + self.row_ptrs.len() * std::mem::size_of::<usize>()
    }

    /// Convert to dense representation
    pub fn to_dense(&self) -> Array2<f64> {
        let mut dense = Array2::zeros((self.n_rows, self.n_cols));

        for rowidx in 0..self.n_rows {
            let start = self.row_ptrs[rowidx];
            let end = self.row_ptrs[rowidx + 1];

            for i in start..end {
                let colidx = self.colindices[i];
                let value = self.values[i];
                dense[[rowidx, colidx]] = value;
            }
        }

        dense
    }

    /// Get a specific row as a sparse vector
    pub fn get_row(&self, rowidx: usize) -> Result<SparseVector> {
        if rowidx >= self.n_rows {
            return Err(TextError::InvalidInput(format!(
                "Row index {} out of bounds for matrix with {} rows",
                rowidx, self.n_rows
            )));
        }

        let start = self.row_ptrs[rowidx];
        let end = self.row_ptrs[rowidx + 1];

        let indices: Vec<usize> = self.colindices[start..end].to_vec();
        let values: Vec<f64> = self.values[start..end].to_vec();

        Ok(SparseVector {
            indices,
            values,
            size: self.n_cols,
        })
    }

    /// Multiply by a dense vector
    pub fn dot(&self, vector: &Array1<f64>) -> Result<Array1<f64>> {
        if vector.len() != self.n_cols {
            return Err(TextError::InvalidInput(format!(
                "Vector dimension {} doesn't match matrix columns {}",
                vector.len(),
                self.n_cols
            )));
        }

        let mut result = Array1::zeros(self.n_rows);

        for rowidx in 0..self.n_rows {
            let start = self.row_ptrs[rowidx];
            let end = self.row_ptrs[rowidx + 1];

            let mut sum = 0.0;
            for i in start..end {
                let colidx = self.colindices[i];
                sum += self.values[i] * vector[colidx];
            }
            result[rowidx] = sum;
        }

        Ok(result)
    }
}

/// Sparse vector representation using index-value pairs
#[derive(Debug, Clone)]
pub struct SparseVector {
    /// Indices of non-zero elements
    indices: Vec<usize>,
    /// Values of non-zero elements
    values: Vec<f64>,
    /// Total size of the vector
    size: usize,
}

impl SparseVector {
    /// Create a new sparse vector
    pub fn new(size: usize) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            size,
        }
    }

    /// Create from a dense vector
    pub fn from_dense(dense: &Array1<f64>) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (idx, &value) in dense.iter().enumerate() {
            if value != 0.0 {
                indices.push(idx);
                values.push(value);
            }
        }

        Self {
            indices,
            values,
            size: dense.len(),
        }
    }

    /// Create from indices and values
    pub fn fromindices_values(indices: Vec<usize>, values: Vec<f64>, size: usize) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "Indices and values must have the same length"
        );
        Self {
            indices,
            values,
            size,
        }
    }

    /// Convert to dense representation
    pub fn to_dense(&self) -> Array1<f64> {
        let mut dense = Array1::zeros(self.size);

        for (&idx, &value) in self.indices.iter().zip(self.values.iter()) {
            dense[idx] = value;
        }

        dense
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get the size of the vector
    pub fn size(&self) -> usize {
        self.size
    }

    /// Compute dot product with another sparse vector
    pub fn dotsparse(&self, other: &SparseVector) -> Result<f64> {
        if self.size != other.size {
            return Err(TextError::InvalidInput(format!(
                "Vector dimensions don't match: {} vs {}",
                self.size, other.size
            )));
        }

        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        // Merge-like algorithm for sorted indices
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Equal => {
                    result += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => {
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    j += 1;
                }
            }
        }

        Ok(result)
    }

    /// Compute L2 norm
    pub fn norm(&self) -> f64 {
        self.values.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Scale by a constant
    pub fn scale(&mut self, scalar: f64) {
        for value in &mut self.values {
            *value *= scalar;
        }
    }

    /// Get the indices of non-zero elements
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get the values of non-zero elements
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Get mutable reference to values
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
}

/// Dictionary of Keys (DOK) format for building sparse matrices
#[derive(Debug, Clone)]
pub struct DokMatrix {
    data: HashMap<(usize, usize), f64>,
    n_rows: usize,
    n_cols: usize,
}

impl DokMatrix {
    /// Create a new DOK matrix
    pub fn new(_n_rows: usize, ncols: usize) -> Self {
        Self {
            data: HashMap::new(),
            n_rows: _n_rows,
            n_cols: ncols,
        }
    }

    /// Set a value at the given position
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(TextError::InvalidInput(format!(
                "Index ({}, {}) out of bounds for matrix shape ({}, {})",
                row, col, self.n_rows, self.n_cols
            )));
        }

        if value != 0.0 {
            self.data.insert((row, col), value);
        } else {
            self.data.remove(&(row, col));
        }

        Ok(())
    }

    /// Get a value at the given position
    pub fn get(&self, row: usize, col: usize) -> f64 {
        *self.data.get(&(row, col)).unwrap_or(&0.0)
    }

    /// Convert to CSR format
    pub fn to_csr(&self) -> CsrMatrix {
        let mut entries: Vec<((usize, usize), f64)> =
            self.data.iter().map(|(&k, &v)| (k, v)).collect();

        // Sort by row, then by column
        entries.sort_by_key(|&((r, c), _)| (r, c));

        let mut values = Vec::with_capacity(entries.len());
        let mut colindices = Vec::with_capacity(entries.len());
        let mut row_ptrs = vec![0];

        let mut current_row = 0;

        for ((row, col), value) in entries {
            while current_row < row {
                row_ptrs.push(values.len());
                current_row += 1;
            }

            values.push(value);
            colindices.push(col);
        }

        while current_row < self.n_rows {
            row_ptrs.push(values.len());
            current_row += 1;
        }

        CsrMatrix {
            values,
            colindices,
            row_ptrs,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
        }
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

/// Builder for efficiently constructing sparse matrices row by row
pub struct SparseMatrixBuilder {
    rows: Vec<SparseVector>,
    n_cols: usize,
}

impl SparseMatrixBuilder {
    /// Create a new builder
    pub fn new(_ncols: usize) -> Self {
        Self {
            rows: Vec::new(),
            n_cols: _ncols,
        }
    }

    /// Add a row to the matrix
    pub fn add_row(&mut self, row: SparseVector) -> Result<()> {
        if row.size() != self.n_cols {
            return Err(TextError::InvalidInput(format!(
                "Row size {} doesn't match expected columns {}",
                row.size(),
                self.n_cols
            )));
        }

        self.rows.push(row);
        Ok(())
    }

    /// Build the final CSR matrix
    pub fn build(self) -> CsrMatrix {
        let n_rows = self.rows.len();
        let mut values = Vec::new();
        let mut colindices = Vec::new();
        let mut row_ptrs = vec![0];

        for row in self.rows {
            values.extend(row.values);
            colindices.extend(row.indices);
            row_ptrs.push(values.len());
        }

        CsrMatrix {
            values,
            colindices,
            row_ptrs,
            n_rows,
            n_cols: self.n_cols,
        }
    }
}

/// COO (Coordinate) sparse matrix format for efficient construction
#[derive(Debug, Clone)]
pub struct CooMatrix {
    /// Row indices
    rowindices: Vec<usize>,
    /// Column indices
    colindices: Vec<usize>,
    /// Values
    values: Vec<f64>,
    /// Number of rows
    n_rows: usize,
    /// Number of columns
    n_cols: usize,
}

impl CooMatrix {
    /// Create a new COO matrix
    pub fn new(_n_rows: usize, ncols: usize) -> Self {
        Self {
            rowindices: Vec::new(),
            colindices: Vec::new(),
            values: Vec::new(),
            n_rows: _n_rows,
            n_cols: ncols,
        }
    }

    /// Add a value to the matrix
    pub fn push(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(TextError::InvalidInput(format!(
                "Index ({}, {}) out of bounds for matrix shape ({}, {})",
                row, col, self.n_rows, self.n_cols
            )));
        }

        if value != 0.0 {
            self.rowindices.push(row);
            self.colindices.push(col);
            self.values.push(value);
        }

        Ok(())
    }

    /// Convert to CSR format (efficient for row operations)
    pub fn to_csr(&self) -> CsrMatrix {
        // Sort by row then column
        let mut indices: Vec<usize> = (0..self.values.len()).collect();
        indices.sort_by_key(|&i| (self.rowindices[i], self.colindices[i]));

        let mut values = Vec::with_capacity(self.values.len());
        let mut colindices = Vec::with_capacity(self.values.len());
        let mut row_ptrs = vec![0];

        let mut current_row = 0;
        for &idx in &indices {
            let row = self.rowindices[idx];

            // Fill row pointers for empty rows
            while current_row < row {
                row_ptrs.push(values.len());
                current_row += 1;
            }

            values.push(self.values[idx]);
            colindices.push(self.colindices[idx]);
        }

        // Fill remaining row pointers
        while row_ptrs.len() <= self.n_rows {
            row_ptrs.push(values.len());
        }

        CsrMatrix {
            values,
            colindices,
            row_ptrs,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
        }
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// CSC (Compressed Sparse Column) format for efficient column operations
#[derive(Debug, Clone)]
pub struct CscMatrix {
    /// Non-zero values
    values: Vec<f64>,
    /// Row indices for each value
    rowindices: Vec<usize>,
    /// Column pointers (cumulative sum of non-zeros per column)
    col_ptrs: Vec<usize>,
    /// Number of rows
    n_rows: usize,
    /// Number of columns
    n_cols: usize,
}

impl CscMatrix {
    /// Create from COO format
    pub fn from_coo(coo: &CooMatrix) -> Self {
        // Sort by column then row
        let mut indices: Vec<usize> = (0..coo.values.len()).collect();
        indices.sort_by_key(|&i| (coo.colindices[i], coo.rowindices[i]));

        let mut values = Vec::with_capacity(coo.values.len());
        let mut rowindices = Vec::with_capacity(coo.values.len());
        let mut col_ptrs = vec![0];

        let mut current_col = 0;
        for &idx in &indices {
            let col = coo.colindices[idx];

            // Fill column pointers for empty columns
            while current_col < col {
                col_ptrs.push(values.len());
                current_col += 1;
            }

            values.push(coo.values[idx]);
            rowindices.push(coo.rowindices[idx]);
        }

        // Fill remaining column pointers
        while col_ptrs.len() <= coo.n_cols {
            col_ptrs.push(values.len());
        }

        Self {
            values,
            rowindices,
            col_ptrs,
            n_rows: coo.n_rows,
            n_cols: coo.n_cols,
        }
    }

    /// Get a column as a sparse vector
    pub fn get_col(&self, colidx: usize) -> Result<SparseVector> {
        if colidx >= self.n_cols {
            return Err(TextError::InvalidInput(format!(
                "Column index {} out of bounds for matrix with {} columns",
                colidx, self.n_cols
            )));
        }

        let start = self.col_ptrs[colidx];
        let end = self.col_ptrs[colidx + 1];

        let indices: Vec<usize> = self.rowindices[start..end].to_vec();
        let values: Vec<f64> = self.values[start..end].to_vec();

        Ok(SparseVector::fromindices_values(
            indices,
            values,
            self.n_rows,
        ))
    }

    /// Get the shape of the matrix
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.values.len() * std::mem::size_of::<f64>()
            + self.rowindices.len() * std::mem::size_of::<usize>()
            + self.col_ptrs.len() * std::mem::size_of::<usize>()
    }

    /// Multiply by a dense vector (A^T * x)
    pub fn dot(&self, vector: &Array1<f64>) -> Result<Array1<f64>> {
        if vector.len() != self.n_rows {
            return Err(TextError::InvalidInput(format!(
                "Vector dimension {} doesn't match matrix rows {}",
                vector.len(),
                self.n_rows
            )));
        }

        let mut result = Array1::zeros(self.n_cols);

        for colidx in 0..self.n_cols {
            let start = self.col_ptrs[colidx];
            let end = self.col_ptrs[colidx + 1];

            let mut sum = 0.0;
            for i in start..end {
                let rowidx = self.rowindices[i];
                sum += self.values[i] * vector[rowidx];
            }
            result[colidx] = sum;
        }

        Ok(result)
    }
}

/// Block sparse matrix for better cache efficiency
#[derive(Debug, Clone)]
pub struct BlockSparseMatrix {
    /// Block size (square blocks)
    _blocksize: usize,
    /// Non-empty blocks stored as dense matrices
    blocks: HashMap<(usize, usize), Array2<f64>>,
    /// Number of rows
    n_rows: usize,
    /// Number of columns
    n_cols: usize,
}

impl BlockSparseMatrix {
    /// Create a new block sparse matrix
    pub fn new(_n_rows: usize, n_cols: usize, _blocksize: usize) -> Self {
        Self {
            _blocksize,
            blocks: HashMap::new(),
            n_rows: _n_rows,
            n_cols,
        }
    }

    /// Set a value in the matrix
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(TextError::InvalidInput(format!(
                "Index ({row}, {col}) out of bounds"
            )));
        }

        let block_row = row / self._blocksize;
        let block_col = col / self._blocksize;
        let local_row = row % self._blocksize;
        let local_col = col % self._blocksize;

        let block = self
            .blocks
            .entry((block_row, block_col))
            .or_insert_with(|| Array2::zeros((self._blocksize, self._blocksize)));

        block[[local_row, local_col]] = value;

        Ok(())
    }

    /// Get a value from the matrix
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.n_rows || col >= self.n_cols {
            return 0.0;
        }

        let block_row = row / self._blocksize;
        let block_col = col / self._blocksize;
        let local_row = row % self._blocksize;
        let local_col = col % self._blocksize;

        self.blocks
            .get(&(block_row, block_col))
            .map(|block| block[[local_row, local_col]])
            .unwrap_or(0.0)
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        self.blocks.len() * self._blocksize * self._blocksize * std::mem::size_of::<f64>()
    }
}

/// Hierarchical sparse matrix using two-level indexing for advanced-sparse data
#[derive(Debug, Clone)]
pub struct HierarchicalSparseMatrix {
    /// Top-level blocks (coarse granularity)
    top_level_blocks: HashMap<(usize, usize), CompressedBlock>,
    /// Block size for top level
    top_blocksize: usize,
    /// Sub-block size within each top-level block
    #[allow(dead_code)]
    sub_blocksize: usize,
    /// Matrix dimensions
    n_rows: usize,
    n_cols: usize,
}

/// Compressed block storing sparse data with RLE compression
#[derive(Debug, Clone)]
pub struct CompressedBlock {
    /// Run-length encoded indices and values
    rle_data: Vec<(usize, f64)>, // (run_length, value)
    /// Original block dimensions
    #[allow(dead_code)]
    block_rows: usize,
    #[allow(dead_code)]
    block_cols: usize,
}

impl HierarchicalSparseMatrix {
    /// Create new hierarchical sparse matrix
    pub fn new(_n_rows: usize, n_cols: usize, top_blocksize: usize, sub_blocksize: usize) -> Self {
        Self {
            top_level_blocks: HashMap::new(),
            top_blocksize,
            sub_blocksize,
            n_rows: _n_rows,
            n_cols,
        }
    }

    /// Set value with hierarchical indexing
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(TextError::InvalidInput(format!(
                "Index ({row}, {col}) out of bounds"
            )));
        }

        let top_row = row / self.top_blocksize;
        let top_col = col / self.top_blocksize;

        // For simplicity, store as COO within each block for now
        // In a full implementation, this would use sub-blocks
        let block = self
            .top_level_blocks
            .entry((top_row, top_col))
            .or_insert_with(|| CompressedBlock::new(self.top_blocksize, self.top_blocksize));

        // Store the linearized position and value
        let local_row = row % self.top_blocksize;
        let local_col = col % self.top_blocksize;
        let linear_pos = local_row * self.top_blocksize + local_col;

        block.set_value(linear_pos, value);
        Ok(())
    }

    /// Get value with hierarchical lookup
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.n_rows || col >= self.n_cols {
            return 0.0;
        }

        let top_row = row / self.top_blocksize;
        let top_col = col / self.top_blocksize;

        if let Some(block) = self.top_level_blocks.get(&(top_row, top_col)) {
            let local_row = row % self.top_blocksize;
            let local_col = col % self.top_blocksize;
            let linear_pos = local_row * self.top_blocksize + local_col;
            block.get_value(linear_pos)
        } else {
            0.0
        }
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        self.top_level_blocks
            .values()
            .map(|block| block.memory_usage())
            .sum::<usize>()
            + std::mem::size_of::<HashMap<(usize, usize), CompressedBlock>>()
    }
}

impl CompressedBlock {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            rle_data: Vec::new(),
            block_rows: rows,
            block_cols: cols,
        }
    }

    fn set_value(&mut self, position: usize, value: f64) {
        if value == 0.0 {
            // Remove zero values to maintain sparsity
            self.rle_data.retain(|(pos_, _)| *pos_ != position);
            return;
        }

        // Find if position already exists
        if let Some(entry) = self.rle_data.iter_mut().find(|(pos_, _)| *pos_ == position) {
            entry.1 = value;
        } else {
            // Insert new entry and keep sorted by position
            let insert_pos = self
                .rle_data
                .binary_search_by_key(&position, |&(pos_, _)| pos_)
                .unwrap_or_else(|pos| pos);
            self.rle_data.insert(insert_pos, (position, value));
        }
    }

    fn get_value(&self, position: usize) -> f64 {
        self.rle_data
            .iter()
            .find(|(pos_, _)| *pos_ == position)
            .map(|(_, value)| *value)
            .unwrap_or(0.0)
    }

    fn memory_usage(&self) -> usize {
        self.rle_data.len() * std::mem::size_of::<(usize, f64)>()
    }
}

/// Bit-packed sparse vector for boolean data with extreme compression
#[derive(Debug, Clone)]
pub struct BitPackedSparseVector {
    /// Bit-packed indices (each bit represents presence/absence)
    bit_data: Vec<u64>,
    /// Total size of the vector
    size: usize,
}

impl BitPackedSparseVector {
    /// Create from a boolean sparse vector
    pub fn from_boolindices(indices: &[usize], size: usize) -> Self {
        let num_words = size.div_ceil(64); // Round up to nearest 64
        let mut bit_data = vec![0u64; num_words];

        for &idx in indices {
            if idx < size {
                let word_idx = idx / 64;
                let bit_idx = idx % 64;
                bit_data[word_idx] |= 1u64 << bit_idx;
            }
        }

        Self { bit_data, size }
    }

    /// Get all set indices
    pub fn getindices(&self) -> Vec<usize> {
        let mut indices = Vec::new();

        for (word_idx, &word) in self.bit_data.iter().enumerate() {
            if word != 0 {
                for bit_idx in 0..64 {
                    if word & (1u64 << bit_idx) != 0 {
                        let global_idx = word_idx * 64 + bit_idx;
                        if global_idx < self.size {
                            indices.push(global_idx);
                        }
                    }
                }
            }
        }

        indices
    }

    /// Check if index is set
    pub fn is_set(&self, idx: usize) -> bool {
        if idx >= self.size {
            return false;
        }

        let word_idx = idx / 64;
        let bit_idx = idx % 64;

        if word_idx < self.bit_data.len() {
            self.bit_data[word_idx] & (1u64 << bit_idx) != 0
        } else {
            false
        }
    }

    /// Count number of set bits
    pub fn count_ones(&self) -> usize {
        self.bit_data
            .iter()
            .map(|&word| word.count_ones() as usize)
            .sum()
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.bit_data.len() * std::mem::size_of::<u64>() + std::mem::size_of::<usize>()
    }

    /// Intersection with another bit-packed vector
    pub fn intersection(&self, other: &BitPackedSparseVector) -> Result<BitPackedSparseVector> {
        if self.size != other.size {
            return Err(TextError::InvalidInput(format!(
                "Vector sizes don't match for intersection: {} vs {}",
                self.size, other.size
            )));
        }

        let mut result_data = Vec::with_capacity(self.bit_data.len());
        for (&a, &b) in self.bit_data.iter().zip(other.bit_data.iter()) {
            result_data.push(a & b);
        }

        Ok(BitPackedSparseVector {
            bit_data: result_data,
            size: self.size,
        })
    }

    /// Union with another bit-packed vector
    pub fn union(&self, other: &BitPackedSparseVector) -> Result<BitPackedSparseVector> {
        if self.size != other.size {
            return Err(TextError::InvalidInput(format!(
                "Vector sizes don't match for union: {} vs {}",
                self.size, other.size
            )));
        }

        let mut result_data = Vec::with_capacity(self.bit_data.len());
        for (&a, &b) in self.bit_data.iter().zip(other.bit_data.iter()) {
            result_data.push(a | b);
        }

        Ok(BitPackedSparseVector {
            bit_data: result_data,
            size: self.size,
        })
    }
}

/// Memory pool for efficient sparse matrix allocation
pub struct SparseMemoryPool {
    /// Pre-allocated f64 values
    value_pool: Vec<Vec<f64>>,
    /// Pre-allocated usize indices
    index_pool: Vec<Vec<usize>>,
    /// Block size for allocations
    _blocksize: usize,
}

impl SparseMemoryPool {
    /// Create new memory pool
    pub fn new(__blocksize: usize) -> Self {
        Self {
            value_pool: Vec::new(),
            index_pool: Vec::new(),
            _blocksize: __blocksize,
        }
    }

    /// Allocate a vector of f64 values
    pub fn allocate_values(&mut self, size: usize) -> Vec<f64> {
        if let Some(mut vec) = self.value_pool.pop() {
            vec.clear();
            vec.reserve(size);
            vec
        } else {
            Vec::with_capacity(size.max(self._blocksize))
        }
    }

    /// Allocate a vector of indices
    pub fn allocateindices(&mut self, size: usize) -> Vec<usize> {
        if let Some(mut vec) = self.index_pool.pop() {
            vec.clear();
            vec.reserve(size);
            vec
        } else {
            Vec::with_capacity(size.max(self._blocksize))
        }
    }

    /// Return vectors to the pool
    pub fn deallocate_values(&mut self, mut vec: Vec<f64>) {
        if vec.capacity() >= self._blocksize {
            vec.clear();
            self.value_pool.push(vec);
        }
    }

    /// Return index vectors to the pool
    pub fn deallocateindices(&mut self, mut vec: Vec<usize>) {
        if vec.capacity() >= self._blocksize {
            vec.clear();
            self.index_pool.push(vec);
        }
    }
}

/// Adaptive sparse matrix that chooses optimal format based on sparsity pattern
#[derive(Debug)]
pub enum AdaptiveSparseMatrix {
    /// Compressed Sparse Row format
    Csr(CsrMatrix),
    /// Compressed Sparse Column format
    Csc(CscMatrix),
    /// Coordinate format
    Coo(CooMatrix),
    /// Block sparse format
    Block(BlockSparseMatrix),
    /// Hierarchical sparse format
    Hierarchical(HierarchicalSparseMatrix),
}

impl AdaptiveSparseMatrix {
    /// Create from COO and automatically select best format
    pub fn from_coo_adaptive(coo: CooMatrix) -> Self {
        let nnz = coo.nnz();
        let (n_rows, n_cols) = (coo.n_rows, coo.n_cols);
        let total_elements = n_rows * n_cols;
        let sparsity = nnz as f64 / total_elements as f64;

        // Choose format based on sparsity and size
        if sparsity < 0.001 && total_elements > 10000 {
            // Very sparse large matrix - use hierarchical
            let hierarchical = HierarchicalSparseMatrix::new(n_rows, n_cols, 64, 8);
            // Convert COO to hierarchical (simplified)
            Self::Hierarchical(hierarchical)
        } else if sparsity < 0.01 && nnz > 1000 {
            // Sparse with some structure - use block format
            let _blocksize = ((nnz as f64).sqrt() as usize).clamp(8, 64);
            Self::Block(BlockSparseMatrix::new(n_rows, n_cols, _blocksize))
        } else if n_rows > n_cols * 2 {
            // Tall matrix - CSC might be better for column operations
            Self::Csc(CscMatrix::from_coo(&coo))
        } else {
            // Default to CSR for row operations
            Self::Csr(coo.to_csr())
        }
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        match self {
            Self::Csr(m) => m.memory_usage(),
            Self::Csc(m) => m.memory_usage(),
            Self::Coo(m) => {
                m.nnz() * (2 * std::mem::size_of::<usize>() + std::mem::size_of::<f64>())
            }
            Self::Block(m) => m.memory_usage(),
            Self::Hierarchical(m) => m.memory_usage(),
        }
    }
}

/// Streaming sparse matrix processor for out-of-core operations
pub struct StreamingSparseProcessor {
    chunk_size: usize,
    #[allow(dead_code)]
    memory_limit: usize,
}

impl StreamingSparseProcessor {
    /// Create new streaming processor
    pub fn new(_chunk_size: usize, memorylimit: usize) -> Self {
        Self {
            chunk_size: _chunk_size,
            memory_limit: memorylimit,
        }
    }

    /// Process large sparse matrix in chunks
    pub fn process_chunks<F, R>(&self, matrix: &CsrMatrix, processor: F) -> Result<Vec<R>>
    where
        F: Fn(&CsrMatrix) -> Result<R>,
    {
        let mut results = Vec::new();
        let (n_rows, n_cols) = matrix.shape();
        let rows_per_chunk = (self.chunk_size).min(n_rows);

        for chunk_start in (0..n_rows).step_by(rows_per_chunk) {
            let chunk_end = (chunk_start + rows_per_chunk).min(n_rows);

            // Extract chunk (simplified - would implement efficient row slicing)
            let chunk = self.extract_row_slice(matrix, chunk_start, chunk_end)?;
            let result = processor(&chunk)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Extract a slice of rows from CSR matrix
    fn extract_row_slice(
        &self,
        matrix: &CsrMatrix,
        start_row: usize,
        end_row: usize,
    ) -> Result<CsrMatrix> {
        // Simplified implementation - would efficiently slice the CSR data
        let mut values = Vec::new();
        let mut colindices = Vec::new();
        let mut row_ptrs = vec![0];

        for _row in start_row..end_row {
            if let Ok(sparse_row) = matrix.get_row(_row) {
                values.extend(sparse_row.values().iter());
                colindices.extend(sparse_row.indices().iter());
                row_ptrs.push(values.len());
            }
        }

        Ok(CsrMatrix {
            values,
            colindices,
            row_ptrs,
            n_rows: end_row - start_row,
            n_cols: matrix.shape().1,
        })
    }
}

/// Approximate sparse matrix using sketching for reduced memory
#[derive(Debug, Clone)]
pub struct ApproximateSparseMatrix {
    /// Hash-based sketches of rows
    row_sketches: Vec<Vec<(u32, f32)>>, // (hash, value) pairs
    /// Sketch size
    #[allow(dead_code)]
    sketchsize: usize,
    /// Original dimensions
    n_rows: usize,
    #[allow(dead_code)]
    n_cols: usize,
}

impl ApproximateSparseMatrix {
    /// Create approximate representation of a sparse matrix
    pub fn from_csr(matrix: &CsrMatrix, sketchsize: usize) -> Self {
        let (n_rows, n_cols) = matrix.shape();
        let mut row_sketches = Vec::with_capacity(n_rows);

        for rowidx in 0..n_rows {
            if let Ok(row) = matrix.get_row(rowidx) {
                let sketch = Self::create_row_sketch(&row, sketchsize);
                row_sketches.push(sketch);
            } else {
                row_sketches.push(Vec::new());
            }
        }

        Self {
            row_sketches,
            sketchsize,
            n_rows,
            n_cols,
        }
    }

    /// Create a hash-based sketch of a sparse row
    fn create_row_sketch(row: &SparseVector, sketchsize: usize) -> Vec<(u32, f32)> {
        let mut sketch = vec![(0u32, 0.0f32); sketchsize];

        for (&idx, &val) in row.indices().iter().zip(row.values().iter()) {
            // Simple hash function for demonstration
            let hash = ((idx as u64 * 2654435761u64) % (sketchsize as u64)) as usize;
            sketch[hash].0 = idx as u32;
            sketch[hash].1 += val as f32;
        }

        // Keep only non-zero sketches
        sketch.into_iter().filter(|(_, val)| *val != 0.0).collect()
    }

    /// Approximate dot product between two row sketches
    pub fn approximate_row_similarity(&self, row1: usize, row2: usize) -> f32 {
        if row1 >= self.n_rows || row2 >= self.n_rows {
            return 0.0;
        }

        let sketch1 = &self.row_sketches[row1];
        let sketch2 = &self.row_sketches[row2];

        let mut similarity = 0.0f32;
        let mut i = 0;
        let mut j = 0;

        // Merge-like algorithm on hash keys
        while i < sketch1.len() && j < sketch2.len() {
            match sketch1[i].0.cmp(&sketch2[j].0) {
                std::cmp::Ordering::Equal => {
                    similarity += sketch1[i].1 * sketch2[j].1;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }

        similarity
    }

    /// Memory usage
    pub fn memory_usage(&self) -> usize {
        self.row_sketches
            .iter()
            .map(|sketch| sketch.len() * std::mem::size_of::<(u32, f32)>())
            .sum::<usize>()
    }
}

/// Quantized sparse vector for reduced memory usage
#[derive(Debug, Clone)]
pub struct QuantizedSparseVector {
    /// Indices of non-zero elements
    indices: Vec<u32>,
    /// Quantized values (8-bit)
    values: Vec<i8>,
    /// Scale factor for dequantization
    scale: f32,
    /// Zero point for dequantization
    zero_point: i8,
    /// Total size
    size: usize,
}

impl QuantizedSparseVector {
    /// Quantize a sparse vector to 8-bit representation
    pub fn fromsparse(sparse: &SparseVector) -> Self {
        if sparse.values.is_empty() {
            return Self {
                indices: Vec::new(),
                values: Vec::new(),
                scale: 1.0,
                zero_point: 0,
                size: sparse.size,
            };
        }

        // Find min and max values
        let min_val = sparse.values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = sparse
            .values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate quantization parameters
        let range = max_val - min_val;
        let scale = (range / 255.0) as f32;
        let zero_point = (-min_val / range * 255.0).round() as i8;

        // Quantize values
        let indices: Vec<u32> = sparse.indices.iter().map(|&i| i as u32).collect();
        let values: Vec<i8> = sparse
            .values
            .iter()
            .map(|&v| {
                let scaled = ((v - min_val) / range * 255.0).round();
                (scaled.clamp(0.0, 255.0) as i16 - 128) as i8 // Center around 0
            })
            .collect();

        Self {
            indices,
            values,
            scale,
            zero_point,
            size: sparse.size,
        }
    }

    /// Dequantize back to a regular sparse vector
    pub fn tosparse(&self) -> SparseVector {
        let indices: Vec<usize> = self.indices.iter().map(|&i| i as usize).collect();
        let values: Vec<f64> = self
            .values
            .iter()
            .map(|&v| {
                let dequantized = (v as f32 + 128.0 - self.zero_point as f32) * self.scale;
                dequantized as f64
            })
            .collect();

        SparseVector::fromindices_values(indices, values, self.size)
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.indices.len() * std::mem::size_of::<u32>()
            + self.values.len() * std::mem::size_of::<i8>()
            + std::mem::size_of::<f32>() * 2
            + std::mem::size_of::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_csr_from_dense() {
        let dense = arr2(&[[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]);

        let csr = CsrMatrix::from_dense(&dense);

        assert_eq!(csr.shape(), (3, 3));
        assert_eq!(csr.nnz(), 5);

        let reconstructed = csr.to_dense();
        assert_eq!(reconstructed, dense);
    }

    #[test]
    fn testsparse_vector() {
        let dense = arr1(&[0.0, 1.0, 0.0, 2.0, 0.0]);
        let sparse = SparseVector::from_dense(&dense);

        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.size(), 5);

        let reconstructed = sparse.to_dense();
        assert_eq!(reconstructed, dense);
    }

    #[test]
    fn testsparse_dot_product() {
        let v1 = SparseVector::from_dense(&arr1(&[1.0, 0.0, 2.0, 0.0]));
        let v2 = SparseVector::from_dense(&arr1(&[0.0, 3.0, 2.0, 0.0]));

        let dot = v1.dotsparse(&v2).unwrap();
        assert_eq!(dot, 4.0); // 1*0 + 0*3 + 2*2 + 0*0 = 4
    }

    #[test]
    fn test_dok_matrix() {
        let mut dok = DokMatrix::new(3, 3);

        dok.set(0, 0, 1.0).unwrap();
        dok.set(1, 1, 2.0).unwrap();
        dok.set(2, 0, 3.0).unwrap();

        assert_eq!(dok.get(0, 0), 1.0);
        assert_eq!(dok.get(0, 1), 0.0);
        assert_eq!(dok.nnz(), 3);

        let csr = dok.to_csr();
        assert_eq!(csr.nnz(), 3);
    }

    #[test]
    fn test_matrix_builder() {
        let mut builder = SparseMatrixBuilder::new(4);

        builder
            .add_row(SparseVector::from_dense(&arr1(&[1.0, 0.0, 2.0, 0.0])))
            .unwrap();
        builder
            .add_row(SparseVector::from_dense(&arr1(&[0.0, 3.0, 0.0, 4.0])))
            .unwrap();

        let matrix = builder.build();
        assert_eq!(matrix.shape(), (2, 4));
        assert_eq!(matrix.nnz(), 4);
    }

    #[test]
    fn test_memory_efficiency() {
        // Create a large sparse matrix (1000x1000 with 1% density)
        let n = 1000;
        let mut dense = Array2::zeros((n, n));

        // Add some random non-zero values
        for i in 0..n / 10 {
            for j in 0..n / 10 {
                dense[[i, j]] = i as f64 + j as f64;
            }
        }

        let sparse = CsrMatrix::from_dense(&dense);

        // Calculate memory savings
        let dense_memory = n * n * std::mem::size_of::<f64>();
        let sparse_memory = sparse.memory_usage();

        println!("Dense memory: {dense_memory} bytes");
        println!("Sparse memory: {sparse_memory} bytes");
        println!(
            "Memory savings: {:.1}%",
            (1.0 - sparse_memory as f64 / dense_memory as f64) * 100.0
        );

        assert!(sparse_memory < dense_memory / 10); // Should use less than 10% of dense memory
    }

    #[test]
    fn test_coo_matrix() {
        let mut coo = CooMatrix::new(3, 3);

        coo.push(0, 0, 1.0).unwrap();
        coo.push(1, 1, 2.0).unwrap();
        coo.push(2, 0, 3.0).unwrap();
        coo.push(1, 2, 4.0).unwrap();

        assert_eq!(coo.nnz(), 4);

        // Convert to CSR
        let csr = coo.to_csr();
        assert_eq!(csr.nnz(), 4);
        assert_eq!(csr.shape(), (3, 3));

        // Verify values
        let dense = csr.to_dense();
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[1, 1]], 2.0);
        assert_eq!(dense[[2, 0]], 3.0);
        assert_eq!(dense[[1, 2]], 4.0);
    }

    #[test]
    fn test_csc_matrix() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 1.0).unwrap();
        coo.push(1, 1, 2.0).unwrap();
        coo.push(2, 0, 3.0).unwrap();

        let csc = CscMatrix::from_coo(&coo);

        // Get column 0
        let col0 = csc.get_col(0).unwrap();
        assert_eq!(col0.nnz(), 2);
        let dense_col0 = col0.to_dense();
        assert_eq!(dense_col0[0], 1.0);
        assert_eq!(dense_col0[2], 3.0);

        // Get column 1
        let col1 = csc.get_col(1).unwrap();
        assert_eq!(col1.nnz(), 1);
        let dense_col1 = col1.to_dense();
        assert_eq!(dense_col1[1], 2.0);
    }

    #[test]
    fn test_blocksparse_matrix() {
        let mut blocksparse = BlockSparseMatrix::new(10, 10, 3);

        // Set some values
        blocksparse.set(0, 0, 1.0).unwrap();
        blocksparse.set(5, 5, 2.0).unwrap();
        blocksparse.set(9, 9, 3.0).unwrap();

        // Get values
        assert_eq!(blocksparse.get(0, 0), 1.0);
        assert_eq!(blocksparse.get(5, 5), 2.0);
        assert_eq!(blocksparse.get(9, 9), 3.0);
        assert_eq!(blocksparse.get(3, 3), 0.0);

        // Check memory usage
        let memory = blocksparse.memory_usage();
        assert!(memory > 0);
    }

    #[test]
    fn test_quantizedsparse_vector() {
        let mut sparse = SparseVector::new(10);
        sparse.indices = vec![1, 3, 7];
        // Use values that should quantize better - avoiding very small values
        sparse.values = vec![100.0, 50.0, 200.0];

        // Quantize
        let quantized = QuantizedSparseVector::fromsparse(&sparse);
        assert_eq!(quantized.indices.len(), 3);
        assert_eq!(quantized.values.len(), 3);

        // Dequantize
        let dequantized = quantized.tosparse();
        assert_eq!(dequantized.nnz(), 3);

        // Check values are approximately preserved
        let orig_values = sparse.values();
        let deq_values = dequantized.values();
        for i in 0..3 {
            let error = (orig_values[i] - deq_values[i]).abs();
            let relative_error = error / orig_values[i].abs();
            // Increase tolerance slightly for quantization
            assert!(relative_error < 0.05); // Less than 5% error
        }

        // Check memory savings
        let orig_memory = sparse.indices.len() * std::mem::size_of::<usize>()
            + sparse.values.len() * std::mem::size_of::<f64>();
        let quantized_memory = quantized.memory_usage();
        assert!(quantized_memory < orig_memory);
    }
}
