//! Block Sparse Row (BSR) matrix format
//!
//! This module provides the BSR matrix format implementation, which is
//! efficient for block-structured matrices.

use crate::error::{SparseError, SparseResult};
use num_traits::Zero;

/// Block Sparse Row (BSR) matrix
///
/// A sparse matrix format that stores blocks in compressed sparse row format,
/// making it efficient for block-structured matrices.
pub struct BsrMatrix<T> {
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Block size (r, c)
    block_size: (usize, usize),
    /// Number of block rows
    block_rows: usize,
    /// Number of block columns (needed for internal calculations)
    #[allow(dead_code)]
    block_cols: usize,
    /// Data array (blocks stored row by row)
    data: Vec<Vec<Vec<T>>>,
    /// Column indices for each block
    indices: Vec<Vec<usize>>,
    /// Row pointers (indptr)
    indptr: Vec<usize>,
}

impl<T> BsrMatrix<T>
where
    T: Clone + Copy + Zero + std::cmp::PartialEq,
{
    /// Create a new BSR matrix
    ///
    /// # Arguments
    ///
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    /// * `block_size` - Tuple containing the block dimensions (r, c)
    ///
    /// # Returns
    ///
    /// * A new empty BSR matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_sparse::bsr::BsrMatrix;
    ///
    /// // Create a 6x6 sparse matrix with 2x2 blocks
    /// let matrix = BsrMatrix::<f64>::new((6, 6), (2, 2)).unwrap();
    /// ```
    pub fn new(shape: (usize, usize), block_size: (usize, usize)) -> SparseResult<Self> {
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
        let block_cols = (cols + c - 1) / c; // Ceiling division

        // Initialize empty BSR matrix
        let data = Vec::new();
        let indices = Vec::new();
        let indptr = vec![0]; // Initial indptr

        Ok(BsrMatrix {
            rows,
            cols,
            block_size,
            block_rows,
            block_cols,
            data,
            indices,
            indptr,
        })
    }

    /// Create a BSR matrix from block data
    ///
    /// # Arguments
    ///
    /// * `data` - Block data (blocks stored row by row)
    /// * `indices` - Column indices for each block
    /// * `indptr` - Row pointers
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    /// * `block_size` - Tuple containing the block dimensions (r, c)
    ///
    /// # Returns
    ///
    /// * A new BSR matrix
    pub fn from_blocks(
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
        let block_cols = (cols + c - 1) / c; // Ceiling division

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

        for &idx in indices.iter().flatten() {
            if idx >= block_cols {
                return Err(SparseError::ValueError(format!(
                    "index {} out of bounds (max {})",
                    idx,
                    block_cols - 1
                )));
            }
        }

        Ok(BsrMatrix {
            rows,
            cols,
            block_size,
            block_rows,
            block_cols,
            data,
            indices,
            indptr,
        })
    }

    /// Get the number of rows in the matrix
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns in the matrix
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the shape (dimensions) of the matrix
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get the block size
    pub fn block_size(&self) -> (usize, usize) {
        self.block_size
    }

    /// Get immutable access to the row pointers (indptr) array
    ///
    /// The indptr array indicates where each block row starts in the indices
    /// and data arrays. Specifically, block row `i` contains blocks
    /// `indptr[i]..indptr[i+1]`.
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Get immutable access to the column indices array
    ///
    /// The indices array contains the block column indices for each stored block.
    /// The indices for block row `i` are stored in
    /// `indices[indptr[i]..indptr[i+1]]`.
    pub fn indices(&self) -> &[Vec<usize>] {
        &self.indices
    }

    /// Get mutable access to the block data array
    ///
    /// The data array contains the non-zero blocks of the matrix, stored in
    /// row-major order. Each block is a 2D array (Vec<Vec<T>>) with dimensions
    /// matching the block size of the matrix.
    pub fn data_mut(&mut self) -> &mut [Vec<Vec<T>>] {
        &mut self.data
    }

    /// Get the number of non-zero blocks in the matrix
    pub fn nnz_blocks(&self) -> usize {
        self.data.len()
    }

    /// Get the number of non-zero elements in the matrix
    pub fn nnz(&self) -> usize {
        // Count non-zeros in all blocks
        let mut count = 0;

        for block in &self.data {
            for row in block {
                for &val in row {
                    if val != T::zero() {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    /// Convert to dense matrix (as Vec<Vec<T>>)
    pub fn to_dense(&self) -> Vec<Vec<T>>
    where
        T: Zero + Copy,
    {
        let mut result = vec![vec![T::zero(); self.cols]; self.rows];
        let (r, c) = self.block_size;

        for block_row in 0..self.block_rows {
            for k in self.indptr[block_row]..self.indptr[block_row + 1] {
                let block_col = self.indices[k][0];
                let block = &self.data[k];

                // Copy block to dense matrix
                for (i, block_row_data) in block.iter().enumerate().take(r) {
                    let row = block_row * r + i;
                    if row < self.rows {
                        for (j, &value) in block_row_data.iter().enumerate().take(c) {
                            let col = block_col * c + j;
                            if col < self.cols {
                                result[row][col] = value;
                            }
                        }
                    }
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bsr_create() {
        // Create a 6x6 sparse matrix with 2x2 blocks
        let matrix = BsrMatrix::<f64>::new((6, 6), (2, 2)).unwrap();

        assert_eq!(matrix.shape(), (6, 6));
        assert_eq!(matrix.block_size(), (2, 2));
        assert_eq!(matrix.nnz_blocks(), 0);
        assert_eq!(matrix.nnz(), 0);
    }

    #[test]
    fn test_bsr_from_blocks() {
        // Create a 4x4 sparse matrix with 2x2 blocks
        // [1 2 0 0]
        // [3 4 0 0]
        // [0 0 5 6]
        // [0 0 7 8]

        let block1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let block2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let data = vec![block1, block2];
        let indices = vec![vec![0], vec![1]];
        let indptr = vec![0, 1, 2];

        let matrix = BsrMatrix::from_blocks(data, indices, indptr, (4, 4), (2, 2)).unwrap();

        assert_eq!(matrix.shape(), (4, 4));
        assert_eq!(matrix.block_size(), (2, 2));
        assert_eq!(matrix.nnz_blocks(), 2);
        assert_eq!(matrix.nnz(), 8); // All elements are non-zero

        // Convert to dense
        let dense = matrix.to_dense();

        let expected = vec![
            vec![1.0, 2.0, 0.0, 0.0],
            vec![3.0, 4.0, 0.0, 0.0],
            vec![0.0, 0.0, 5.0, 6.0],
            vec![0.0, 0.0, 7.0, 8.0],
        ];

        assert_eq!(dense, expected);
    }
}
