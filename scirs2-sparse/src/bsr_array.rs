// BSR Array implementation
//
// This module provides the BSR (Block Sparse Row) array format,
// which is efficient for matrices with block-structured sparsity patterns.

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use std::fmt::{self, Debug};
use std::ops::{Add, Div, Mul, Sub};

use crate::coo_array::CooArray;
use crate::csc_array::CscArray;
use crate::csr_array::CsrArray;
use crate::dia_array::DiaArray;
use crate::dok_array::DokArray;
use crate::error::{SparseError, SparseResult};
use crate::lil_array::LilArray;
use crate::sparray::{SparseArray, SparseSum};

/// BSR Array format
///
/// The BSR (Block Sparse Row) format stores a sparse matrix as a sparse matrix
/// of dense blocks. It's particularly efficient for matrices with block-structured
/// sparsity patterns, such as those arising in finite element methods.
///
/// # Notes
///
/// - Very efficient for matrices with block structure
/// - Fast matrix-vector products for block-structured matrices
/// - Reduced indexing overhead compared to CSR for block-structured problems
/// - Not efficient for general sparse matrices
/// - Difficult to modify once constructed
///
#[derive(Clone)]
pub struct BsrArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static
        + std::ops::AddAssign,
{
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Block size (r, c)
    block_size: (usize, usize),
    /// Number of block rows
    block_rows: usize,
    /// Number of block columns (needed for internal calculations)
    _block_cols: usize,
    /// Data array (blocks stored row by row)
    data: Vec<Vec<Vec<T>>>,
    /// Column indices for each block
    indices: Vec<Vec<usize>>,
    /// Row pointers (indptr)
    indptr: Vec<usize>,
}

impl<T> BsrArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static
        + std::ops::AddAssign,
{
    /// Create a new BSR array from raw data
    ///
    /// # Arguments
    ///
    /// * `data` - Block data (blocks stored row by row)
    /// * `indices` - Column indices for each block
    /// * `indptr` - Row pointers
    /// * `shape` - Tuple containing the array dimensions (rows, cols)
    /// * `block_size` - Tuple containing the block dimensions (r, c)
    ///
    /// # Returns
    ///
    /// * A new BSR array
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_sparse::bsr_array::BsrArray;
    /// use scirs2_sparse::sparray::SparseArray;
    ///
    /// // Create a 4x4 sparse array with 2x2 blocks
    /// // [1 2 0 0]
    /// // [3 4 0 0]
    /// // [0 0 5 6]
    /// // [0 0 7 8]
    ///
    /// let block1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    /// let block2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    ///
    /// let data = vec![block1, block2];
    /// let indices = vec![vec![0], vec![1]];
    /// let indptr = vec![0, 1, 2];
    ///
    /// let array = BsrArray::new(data, indices, indptr, (4, 4), (2, 2)).unwrap();
    /// assert_eq!(array.shape(), (4, 4));
    /// assert_eq!(array.nnz(), 8); // All elements in the blocks are non-zero
    /// ```
    pub fn new(
        data: Vec<Vec<Vec<T>>>,
        indices: Vec<Vec<usize>>,
        indptr: Vec<usize>,
        shape: (usize, usize),
        block_size: (usize, usize),
    ) -> SparseResult<Self> {
        let (rows, cols) = shape;
        let (r, c) = block_size;

        if r == 0 || c == 0 {
            return Err(SparseError::ValueError(
                "Block dimensions must be positive".to_string(),
            ));
        }

        // Calculate block dimensions
        #[allow(clippy::manual_div_ceil)]
        let block_rows = (rows + r - 1) / r; // Ceiling division
        #[allow(clippy::manual_div_ceil)]
        let _block_cols = (cols + c - 1) / c; // Ceiling division

        // Validate input
        if indptr.len() != block_rows + 1 {
            return Err(SparseError::DimensionMismatch {
                expected: block_rows + 1,
                found: indptr.len(),
            });
        }

        if data.len() != indptr[block_rows] {
            return Err(SparseError::DimensionMismatch {
                expected: indptr[block_rows],
                found: data.len(),
            });
        }

        if indices.len() != data.len() {
            return Err(SparseError::DimensionMismatch {
                expected: data.len(),
                found: indices.len(),
            });
        }

        for block in data.iter() {
            if block.len() != r {
                return Err(SparseError::DimensionMismatch {
                    expected: r,
                    found: block.len(),
                });
            }

            for row in block.iter() {
                if row.len() != c {
                    return Err(SparseError::DimensionMismatch {
                        expected: c,
                        found: row.len(),
                    });
                }
            }
        }

        for idx_vec in indices.iter() {
            if idx_vec.len() != 1 {
                return Err(SparseError::ValueError(
                    "Each index vector must contain exactly one block column index".to_string(),
                ));
            }
            if idx_vec[0] >= _block_cols {
                return Err(SparseError::ValueError(format!(
                    "index {} out of bounds (max {})",
                    idx_vec[0],
                    _block_cols - 1
                )));
            }
        }

        Ok(BsrArray {
            rows,
            cols,
            block_size,
            block_rows,
            _block_cols,
            data,
            indices,
            indptr,
        })
    }

    /// Create a new empty BSR array
    ///
    /// # Arguments
    ///
    /// * `shape` - Tuple containing the array dimensions (rows, cols)
    /// * `block_size` - Tuple containing the block dimensions (r, c)
    ///
    /// # Returns
    ///
    /// * A new empty BSR array
    pub fn empty(shape: (usize, usize), block_size: (usize, usize)) -> SparseResult<Self> {
        let (rows, cols) = shape;
        let (r, c) = block_size;

        if r == 0 || c == 0 {
            return Err(SparseError::ValueError(
                "Block dimensions must be positive".to_string(),
            ));
        }

        // Calculate block dimensions
        #[allow(clippy::manual_div_ceil)]
        let block_rows = (rows + r - 1) / r; // Ceiling division
        #[allow(clippy::manual_div_ceil)]
        let _block_cols = (cols + c - 1) / c; // Ceiling division

        // Initialize empty BSR array
        let data = Vec::new();
        let indices = Vec::new();
        let indptr = vec![0; block_rows + 1];

        Ok(BsrArray {
            rows,
            cols,
            block_size,
            block_rows,
            _block_cols,
            data,
            indices,
            indptr,
        })
    }

    /// Convert triplets to BSR format
    ///
    /// # Arguments
    ///
    /// * `row` - Row indices
    /// * `col` - Column indices
    /// * `data` - Data values
    /// * `shape` - Shape of the array
    /// * `block_size` - Size of the blocks
    ///
    /// # Returns
    ///
    /// * A new BSR array
    pub fn from_triplets(
        row: &[usize],
        col: &[usize],
        data: &[T],
        shape: (usize, usize),
        block_size: (usize, usize),
    ) -> SparseResult<Self> {
        if row.len() != col.len() || row.len() != data.len() {
            return Err(SparseError::InconsistentData {
                reason: "Lengths of row, col, and data arrays must be equal".to_string(),
            });
        }

        let (rows, cols) = shape;
        let (r, c) = block_size;

        if r == 0 || c == 0 {
            return Err(SparseError::ValueError(
                "Block dimensions must be positive".to_string(),
            ));
        }

        // Calculate block dimensions
        #[allow(clippy::manual_div_ceil)]
        let block_rows = (rows + r - 1) / r; // Ceiling division
        #[allow(clippy::manual_div_ceil)]
        let _block_cols = (cols + c - 1) / c; // Ceiling division

        // First, we'll construct a temporary DOK-like representation for the blocks
        let mut block_data = std::collections::HashMap::new();

        // Assign each element to its corresponding block
        for (&row_idx, (&col_idx, &val)) in row.iter().zip(col.iter().zip(data.iter())) {
            if row_idx >= rows || col_idx >= cols {
                return Err(SparseError::IndexOutOfBounds {
                    index: (row_idx, col_idx),
                    shape,
                });
            }

            // Calculate block indices
            let block_row = row_idx / r;
            let block_col = col_idx / c;

            // Calculate position within block
            let block_row_pos = row_idx % r;
            let block_col_pos = col_idx % c;

            // Create or get the block
            let block = block_data.entry((block_row, block_col)).or_insert_with(|| {
                let block = vec![vec![T::zero(); c]; r];
                block
            });

            // Set the value in the block
            block[block_row_pos][block_col_pos] = val;
        }

        // Now convert the DOK-like format to BSR
        let mut rows_with_blocks: Vec<usize> = block_data.keys().map(|&(row, _)| row).collect();
        rows_with_blocks.sort();
        rows_with_blocks.dedup();

        // Create indptr array
        let mut indptr = vec![0; block_rows + 1];
        let mut current_nnz = 0;

        // Sorted blocks data and indices
        let mut data = Vec::new();
        let mut indices = Vec::new();

        for row_idx in 0..block_rows {
            if rows_with_blocks.contains(&row_idx) {
                // Get all blocks for this row
                let mut row_blocks: Vec<(usize, Vec<Vec<T>>)> = block_data
                    .iter()
                    .filter(|&(&(r, _), _)| r == row_idx)
                    .map(|(&(_, c), block)| (c, block.clone()))
                    .collect();

                // Sort by column index
                row_blocks.sort_by_key(|&(col, _)| col);

                // Add to data and indices
                for (col, block) in row_blocks {
                    data.push(block);
                    indices.push(vec![col]);
                    current_nnz += 1;
                }
            }

            indptr[row_idx + 1] = current_nnz;
        }

        // Create the BSR array
        BsrArray::new(data, indices, indptr, shape, block_size)
    }

    /// Convert to COO format triplets
    fn to_coo_internal(&self) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        let (r, c) = self.block_size;
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for block_row in 0..self.block_rows {
            for k in self.indptr[block_row]..self.indptr[block_row + 1] {
                let block_col = self.indices[k][0];
                let block = &self.data[k];

                // For each element in the block
                for (i, block_row_data) in block.iter().enumerate().take(r) {
                    let row = block_row * r + i;
                    if row < self.rows {
                        for (j, &value) in block_row_data.iter().enumerate().take(c) {
                            let col = block_col * c + j;
                            if col < self.cols && !value.is_zero() {
                                row_indices.push(row);
                                col_indices.push(col);
                                values.push(value);
                            }
                        }
                    }
                }
            }
        }

        (row_indices, col_indices, values)
    }
}

impl<T> SparseArray<T> for BsrArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static
        + std::ops::AddAssign,
{
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn nnz(&self) -> usize {
        let mut count = 0;

        for block in &self.data {
            for row in block {
                for &val in row {
                    if !val.is_zero() {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    fn dtype(&self) -> &str {
        "float" // Placeholder; ideally would return the actual type
    }

    fn to_array(&self) -> Array2<T> {
        let mut result = Array2::zeros((self.rows, self.cols));
        let (r, c) = self.block_size;

        for block_row in 0..self.block_rows {
            for k in self.indptr[block_row]..self.indptr[block_row + 1] {
                let block_col = self.indices[k][0];
                let block = &self.data[k];

                // Copy block to dense array
                for (i, block_row_data) in block.iter().enumerate().take(r) {
                    let row = block_row * r + i;
                    if row < self.rows {
                        for (j, &value) in block_row_data.iter().enumerate().take(c) {
                            let col = block_col * c + j;
                            if col < self.cols {
                                result[[row, col]] = value;
                            }
                        }
                    }
                }
            }
        }

        result
    }

    fn toarray(&self) -> Array2<T> {
        self.to_array()
    }

    fn to_coo(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (row_indices, col_indices, values) = self.to_coo_internal();
        CooArray::from_triplets(
            &row_indices,
            &col_indices,
            &values,
            (self.rows, self.cols),
            false,
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_csr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (row_indices, col_indices, values) = self.to_coo_internal();
        CsrArray::from_triplets(
            &row_indices,
            &col_indices,
            &values,
            (self.rows, self.cols),
            false,
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_csc(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (row_indices, col_indices, values) = self.to_coo_internal();
        CscArray::from_triplets(
            &row_indices,
            &col_indices,
            &values,
            (self.rows, self.cols),
            false,
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_dok(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (row_indices, col_indices, values) = self.to_coo_internal();
        DokArray::from_triplets(&row_indices, &col_indices, &values, (self.rows, self.cols))
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_lil(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (row_indices, col_indices, values) = self.to_coo_internal();
        LilArray::from_triplets(&row_indices, &col_indices, &values, (self.rows, self.cols))
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_dia(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (row_indices, col_indices, values) = self.to_coo_internal();
        DiaArray::from_triplets(&row_indices, &col_indices, &values, (self.rows, self.cols))
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_bsr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        Ok(Box::new(self.clone()))
    }

    fn add(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert both to CSR for addition
        let csr_self = self.to_csr()?;
        let csr_other = other.to_csr()?;
        csr_self.add(&*csr_other)
    }

    fn sub(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert both to CSR for subtraction
        let csr_self = self.to_csr()?;
        let csr_other = other.to_csr()?;
        csr_self.sub(&*csr_other)
    }

    fn mul(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert both to CSR for element-wise multiplication
        let csr_self = self.to_csr()?;
        let csr_other = other.to_csr()?;
        csr_self.mul(&*csr_other)
    }

    fn div(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert both to CSR for element-wise division
        let csr_self = self.to_csr()?;
        let csr_other = other.to_csr()?;
        csr_self.div(&*csr_other)
    }

    fn dot(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (_, n) = self.shape();
        let (p, q) = other.shape();

        if n != p {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: p,
            });
        }

        // If other is a vector (thin matrix), we can use optimized BSR-Vector multiplication
        if q == 1 {
            // Get the vector from other
            let other_array = other.to_array();
            let vec_view = other_array.column(0);

            // Perform BSR-Vector multiplication
            let result = self.dot_vector(&vec_view)?;

            // Convert to a matrix - create a COO from triplets
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            let mut values = Vec::new();

            for (i, &val) in result.iter().enumerate() {
                if !val.is_zero() {
                    rows.push(i);
                    cols.push(0);
                    values.push(val);
                }
            }

            CooArray::from_triplets(&rows, &cols, &values, (result.len(), 1), false)
                .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
        } else {
            // For general matrix-matrix multiplication, convert to CSR
            let csr_self = self.to_csr()?;
            csr_self.dot(other)
        }
    }

    fn dot_vector(&self, other: &ArrayView1<T>) -> SparseResult<Array1<T>> {
        let (rows, cols) = self.shape();
        let (r, c) = self.block_size;

        if cols != other.len() {
            return Err(SparseError::DimensionMismatch {
                expected: cols,
                found: other.len(),
            });
        }

        let mut result = Array1::zeros(rows);

        for block_row in 0..self.block_rows {
            for k in self.indptr[block_row]..self.indptr[block_row + 1] {
                let block_col = self.indices[k][0];
                let block = &self.data[k];

                // For each element in the block
                for (i, block_row_data) in block.iter().enumerate().take(r) {
                    let row = block_row * r + i;
                    if row < self.rows {
                        for (j, &value) in block_row_data.iter().enumerate().take(c) {
                            let col = block_col * c + j;
                            if col < self.cols {
                                result[row] += value * other[col];
                            }
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    fn transpose(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert to COO, transpose, then convert back to BSR
        self.to_coo()?.transpose()?.to_bsr()
    }

    fn copy(&self) -> Box<dyn SparseArray<T>> {
        Box::new(self.clone())
    }

    fn get(&self, i: usize, j: usize) -> T {
        if i >= self.rows || j >= self.cols {
            return T::zero();
        }

        let (r, c) = self.block_size;
        let block_row = i / r;
        let block_col = j / c;
        let block_row_pos = i % r;
        let block_col_pos = j % c;

        // Search for the block in the row
        for k in self.indptr[block_row]..self.indptr[block_row + 1] {
            if self.indices[k][0] == block_col {
                return self.data[k][block_row_pos][block_col_pos];
            }
        }

        T::zero()
    }

    fn set(&mut self, i: usize, j: usize, value: T) -> SparseResult<()> {
        if i >= self.rows || j >= self.cols {
            return Err(SparseError::IndexOutOfBounds {
                index: (i, j),
                shape: (self.rows, self.cols),
            });
        }

        let (r, c) = self.block_size;
        let block_row = i / r;
        let block_col = j / c;
        let block_row_pos = i % r;
        let block_col_pos = j % c;

        // Search for the block in the row
        for k in self.indptr[block_row]..self.indptr[block_row + 1] {
            if self.indices[k][0] == block_col {
                // Block exists, update value
                self.data[k][block_row_pos][block_col_pos] = value;
                return Ok(());
            }
        }

        // Block doesn't exist, we need to create it
        if !value.is_zero() {
            // Find position to insert
            let pos = self.indptr[block_row + 1];

            // Create new block
            let mut block = vec![vec![T::zero(); c]; r];
            block[block_row_pos][block_col_pos] = value;

            // Insert block, indices
            self.data.insert(pos, block);
            self.indices.insert(pos, vec![block_col]);

            // Update indptr for subsequent rows
            for k in (block_row + 1)..=self.block_rows {
                self.indptr[k] += 1;
            }

            Ok(())
        } else {
            // If value is zero and block doesn't exist, do nothing
            Ok(())
        }
    }

    fn eliminate_zeros(&mut self) {
        // No need to use block_size variables here
        let mut new_data = Vec::new();
        let mut new_indices = Vec::new();
        let mut new_indptr = vec![0];
        let mut current_nnz = 0;

        for block_row in 0..self.block_rows {
            for k in self.indptr[block_row]..self.indptr[block_row + 1] {
                let block_col = self.indices[k][0];
                let block = &self.data[k];

                // Check if block has any non-zero elements
                let mut has_nonzero = false;
                for row in block {
                    for &val in row {
                        if !val.is_zero() {
                            has_nonzero = true;
                            break;
                        }
                    }
                    if has_nonzero {
                        break;
                    }
                }

                if has_nonzero {
                    new_data.push(block.clone());
                    new_indices.push(vec![block_col]);
                    current_nnz += 1;
                }
            }

            new_indptr.push(current_nnz);
        }

        self.data = new_data;
        self.indices = new_indices;
        self.indptr = new_indptr;
    }

    fn sort_indices(&mut self) {
        // No need to use block_size variables here
        let mut new_data = Vec::new();
        let mut new_indices = Vec::new();
        let mut new_indptr = vec![0];
        let mut current_nnz = 0;

        for block_row in 0..self.block_rows {
            // Get blocks for this row
            let mut row_blocks = Vec::new();
            for k in self.indptr[block_row]..self.indptr[block_row + 1] {
                row_blocks.push((self.indices[k][0], self.data[k].clone()));
            }

            // Sort by column index
            row_blocks.sort_by_key(|&(col, _)| col);

            // Add sorted blocks to new data structures
            for (col, block) in row_blocks {
                new_data.push(block);
                new_indices.push(vec![col]);
                current_nnz += 1;
            }

            new_indptr.push(current_nnz);
        }

        self.data = new_data;
        self.indices = new_indices;
        self.indptr = new_indptr;
    }

    fn sorted_indices(&self) -> Box<dyn SparseArray<T>> {
        let mut result = self.clone();
        result.sort_indices();
        Box::new(result)
    }

    fn has_sorted_indices(&self) -> bool {
        for block_row in 0..self.block_rows {
            let mut prev_col = None;

            for k in self.indptr[block_row]..self.indptr[block_row + 1] {
                let col = self.indices[k][0];

                if let Some(prev) = prev_col {
                    if col <= prev {
                        return false;
                    }
                }

                prev_col = Some(col);
            }
        }

        true
    }

    fn sum(&self, axis: Option<usize>) -> SparseResult<SparseSum<T>> {
        match axis {
            None => {
                // Sum all elements
                let mut total = T::zero();

                for block in &self.data {
                    for row in block {
                        for &val in row {
                            total += val;
                        }
                    }
                }

                Ok(SparseSum::Scalar(total))
            }
            Some(0) => {
                // Sum along rows (result is 1 x cols)
                let mut result = vec![T::zero(); self.cols];
                let (r, c) = self.block_size;

                for block_row in 0..self.block_rows {
                    for k in self.indptr[block_row]..self.indptr[block_row + 1] {
                        let block_col = self.indices[k][0];
                        let block = &self.data[k];

                        for block_row_data in block.iter().take(r) {
                            for (j, &value) in block_row_data.iter().enumerate().take(c) {
                                let col = block_col * c + j;
                                if col < self.cols {
                                    result[col] += value;
                                }
                            }
                        }
                    }
                }

                // Create a sparse array from the result
                let mut row_indices = Vec::new();
                let mut col_indices = Vec::new();
                let mut values = Vec::new();

                for (j, &val) in result.iter().enumerate() {
                    if !val.is_zero() {
                        row_indices.push(0);
                        col_indices.push(j);
                        values.push(val);
                    }
                }

                match CooArray::from_triplets(
                    &row_indices,
                    &col_indices,
                    &values,
                    (1, self.cols),
                    false,
                ) {
                    Ok(array) => Ok(SparseSum::SparseArray(Box::new(array))),
                    Err(e) => Err(e),
                }
            }
            Some(1) => {
                // Sum along columns (result is rows x 1)
                let mut result = vec![T::zero(); self.rows];
                let (r, c) = self.block_size;

                for block_row in 0..self.block_rows {
                    for k in self.indptr[block_row]..self.indptr[block_row + 1] {
                        let block = &self.data[k];

                        for (i, block_row_data) in block.iter().enumerate().take(r) {
                            let row = block_row * r + i;
                            if row < self.rows {
                                for &value in block_row_data.iter().take(c) {
                                    result[row] += value;
                                }
                            }
                        }
                    }
                }

                // Create a sparse array from the result
                let mut row_indices = Vec::new();
                let mut col_indices = Vec::new();
                let mut values = Vec::new();

                for (i, &val) in result.iter().enumerate() {
                    if !val.is_zero() {
                        row_indices.push(i);
                        col_indices.push(0);
                        values.push(val);
                    }
                }

                match CooArray::from_triplets(
                    &row_indices,
                    &col_indices,
                    &values,
                    (self.rows, 1),
                    false,
                ) {
                    Ok(array) => Ok(SparseSum::SparseArray(Box::new(array))),
                    Err(e) => Err(e),
                }
            }
            _ => Err(SparseError::InvalidAxis),
        }
    }

    fn max(&self) -> T {
        let mut max_val = T::neg_infinity();

        for block in &self.data {
            for row in block {
                for &val in row {
                    max_val = max_val.max(val);
                }
            }
        }

        // If no elements or all negative infinity, return zero
        if max_val == T::neg_infinity() {
            T::zero()
        } else {
            max_val
        }
    }

    fn min(&self) -> T {
        let mut min_val = T::infinity();
        let mut has_nonzero = false;

        for block in &self.data {
            for row in block {
                for &val in row {
                    if !val.is_zero() {
                        has_nonzero = true;
                        min_val = min_val.min(val);
                    }
                }
            }
        }

        // If no non-zero elements, return zero
        if !has_nonzero {
            T::zero()
        } else {
            min_val
        }
    }

    fn find(&self) -> (Array1<usize>, Array1<usize>, Array1<T>) {
        let (row_indices, col_indices, values) = self.to_coo_internal();

        (
            Array1::from_vec(row_indices),
            Array1::from_vec(col_indices),
            Array1::from_vec(values),
        )
    }

    fn slice(
        &self,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (start_row, end_row) = row_range;
        let (start_col, end_col) = col_range;
        let (rows, cols) = self.shape();

        if start_row >= rows || end_row > rows || start_col >= cols || end_col > cols {
            return Err(SparseError::IndexOutOfBounds {
                index: (start_row.max(end_row), start_col.max(end_col)),
                shape: (rows, cols),
            });
        }

        if start_row >= end_row || start_col >= end_col {
            return Err(SparseError::InvalidSliceRange);
        }

        // Convert to COO, slice, then convert back to BSR
        let coo = self.to_coo()?;
        coo.slice(row_range, col_range)?.to_bsr()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Implement Display for BsrArray for better debugging
impl<T> fmt::Display for BsrArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static
        + std::ops::AddAssign,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "BsrArray of shape {:?} with {} stored elements",
            (self.rows, self.cols),
            self.nnz()
        )?;
        writeln!(f, "Block size: {:?}", self.block_size)?;
        writeln!(f, "Number of blocks: {}", self.data.len())?;

        if self.data.len() <= 5 {
            for block_row in 0..self.block_rows {
                for k in self.indptr[block_row]..self.indptr[block_row + 1] {
                    let block_col = self.indices[k][0];
                    let block = &self.data[k];

                    writeln!(f, "Block at ({}, {}): ", block_row, block_col)?;
                    for row in block {
                        write!(f, "  [")?;
                        for (j, &val) in row.iter().enumerate() {
                            if j > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{:?}", val)?;
                        }
                        writeln!(f, "]")?;
                    }
                }
            }
        } else {
            writeln!(f, "({} blocks total)", self.data.len())?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bsr_array_create() {
        // Create a 4x4 sparse array with 2x2 blocks
        // [1 2 0 0]
        // [3 4 0 0]
        // [0 0 5 6]
        // [0 0 7 8]

        let block1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let block2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let data = vec![block1, block2];
        let indices = vec![vec![0], vec![1]];
        let indptr = vec![0, 1, 2];

        let array = BsrArray::new(data, indices, indptr, (4, 4), (2, 2)).unwrap();

        assert_eq!(array.shape(), (4, 4));
        assert_eq!(array.block_size, (2, 2));
        assert_eq!(array.nnz(), 8); // All elements in the blocks are non-zero

        // Test values
        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(0, 1), 2.0);
        assert_eq!(array.get(1, 0), 3.0);
        assert_eq!(array.get(1, 1), 4.0);
        assert_eq!(array.get(2, 2), 5.0);
        assert_eq!(array.get(2, 3), 6.0);
        assert_eq!(array.get(3, 2), 7.0);
        assert_eq!(array.get(3, 3), 8.0);
        assert_eq!(array.get(0, 2), 0.0); // zero element
    }

    #[test]
    fn test_bsr_array_from_triplets() {
        // Create a 4x4 sparse array with 2x2 blocks
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![0, 1, 0, 1, 2, 3, 2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = (4, 4);
        let block_size = (2, 2);

        let array = BsrArray::from_triplets(&rows, &cols, &data, shape, block_size).unwrap();

        assert_eq!(array.shape(), (4, 4));
        assert_eq!(array.block_size, (2, 2));
        assert_eq!(array.nnz(), 8);

        // Test values
        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(0, 1), 2.0);
        assert_eq!(array.get(1, 0), 3.0);
        assert_eq!(array.get(1, 1), 4.0);
        assert_eq!(array.get(2, 2), 5.0);
        assert_eq!(array.get(2, 3), 6.0);
        assert_eq!(array.get(3, 2), 7.0);
        assert_eq!(array.get(3, 3), 8.0);
        assert_eq!(array.get(0, 2), 0.0); // zero element
    }

    #[test]
    fn test_bsr_array_conversion() {
        // Create a 4x4 sparse array with 2x2 blocks
        let block1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let block2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let data = vec![block1, block2];
        let indices = vec![vec![0], vec![1]];
        let indptr = vec![0, 1, 2];

        let array = BsrArray::new(data, indices, indptr, (4, 4), (2, 2)).unwrap();

        // Convert to COO and check
        let coo = array.to_coo().unwrap();
        assert_eq!(coo.shape(), (4, 4));
        assert_eq!(coo.nnz(), 8);

        // Convert to CSR and check
        let csr = array.to_csr().unwrap();
        assert_eq!(csr.shape(), (4, 4));
        assert_eq!(csr.nnz(), 8);

        // Convert to dense and check
        let dense = array.to_array();
        let expected = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0,
            ],
        )
        .unwrap();
        assert_eq!(dense, expected);
    }

    #[test]
    fn test_bsr_array_operations() {
        // Create two simple block arrays
        let block1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let data1 = vec![block1];
        let indices1 = vec![vec![0]];
        let indptr1 = vec![0, 1];
        let array1 = BsrArray::new(data1, indices1, indptr1, (2, 2), (2, 2)).unwrap();

        let block2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let data2 = vec![block2];
        let indices2 = vec![vec![0]];
        let indptr2 = vec![0, 1];
        let array2 = BsrArray::new(data2, indices2, indptr2, (2, 2), (2, 2)).unwrap();

        // Test addition
        let sum = array1.add(&array2).unwrap();
        assert_eq!(sum.shape(), (2, 2));
        assert_eq!(sum.get(0, 0), 6.0); // 1+5
        assert_eq!(sum.get(0, 1), 8.0); // 2+6
        assert_eq!(sum.get(1, 0), 10.0); // 3+7
        assert_eq!(sum.get(1, 1), 12.0); // 4+8

        // Test element-wise multiplication
        let product = array1.mul(&array2).unwrap();
        assert_eq!(product.shape(), (2, 2));
        assert_eq!(product.get(0, 0), 5.0); // 1*5
        assert_eq!(product.get(0, 1), 12.0); // 2*6
        assert_eq!(product.get(1, 0), 21.0); // 3*7
        assert_eq!(product.get(1, 1), 32.0); // 4*8

        // Test dot product (matrix multiplication)
        let dot = array1.dot(&array2).unwrap();
        assert_eq!(dot.shape(), (2, 2));
        assert_eq!(dot.get(0, 0), 19.0); // 1*5 + 2*7
        assert_eq!(dot.get(0, 1), 22.0); // 1*6 + 2*8
        assert_eq!(dot.get(1, 0), 43.0); // 3*5 + 4*7
        assert_eq!(dot.get(1, 1), 50.0); // 3*6 + 4*8
    }

    #[test]
    fn test_bsr_array_dot_vector() {
        // Create a 4x4 sparse array with 2x2 blocks
        let block1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let block2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let data = vec![block1, block2];
        let indices = vec![vec![0], vec![1]];
        let indptr = vec![0, 1, 2];

        let array = BsrArray::new(data, indices, indptr, (4, 4), (2, 2)).unwrap();

        // Create a vector
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        // Test matrix-vector multiplication
        let result = array.dot_vector(&vector.view()).unwrap();

        // Expected: [1*1 + 2*2 + 0*3 + 0*4, 3*1 + 4*2 + 0*3 + 0*4,
        //            0*1 + 0*2 + 5*3 + 6*4, 0*1 + 0*2 + 7*3 + 8*4]
        // = [5, 11, 39, 53]
        let expected = Array1::from_vec(vec![5.0, 11.0, 39.0, 53.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bsr_array_sum() {
        // Create a 4x4 sparse array with 2x2 blocks
        let block1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let block2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let data = vec![block1, block2];
        let indices = vec![vec![0], vec![1]];
        let indptr = vec![0, 1, 2];

        let array = BsrArray::new(data, indices, indptr, (4, 4), (2, 2)).unwrap();

        // Test sum of entire array
        if let SparseSum::Scalar(sum) = array.sum(None).unwrap() {
            assert_eq!(sum, 36.0); // 1+2+3+4+5+6+7+8 = 36
        } else {
            panic!("Expected SparseSum::Scalar");
        }

        // Test sum along rows (result should be 1 x 4)
        if let SparseSum::SparseArray(row_sum) = array.sum(Some(0)).unwrap() {
            assert_eq!(row_sum.shape(), (1, 4));
            assert_eq!(row_sum.get(0, 0), 4.0); // 1+3
            assert_eq!(row_sum.get(0, 1), 6.0); // 2+4
            assert_eq!(row_sum.get(0, 2), 12.0); // 5+7
            assert_eq!(row_sum.get(0, 3), 14.0); // 6+8
        } else {
            panic!("Expected SparseSum::SparseArray");
        }

        // Test sum along columns (result should be 4 x 1)
        if let SparseSum::SparseArray(col_sum) = array.sum(Some(1)).unwrap() {
            assert_eq!(col_sum.shape(), (4, 1));
            assert_eq!(col_sum.get(0, 0), 3.0); // 1+2
            assert_eq!(col_sum.get(1, 0), 7.0); // 3+4
            assert_eq!(col_sum.get(2, 0), 11.0); // 5+6
            assert_eq!(col_sum.get(3, 0), 15.0); // 7+8
        } else {
            panic!("Expected SparseSum::SparseArray");
        }
    }
}
