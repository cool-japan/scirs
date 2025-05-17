//! Compressed Sparse Row (CSR) matrix format
//!
//! This module provides the CSR matrix format implementation, which is
//! efficient for row operations, matrix-vector multiplication, and more.

use crate::error::{SparseError, SparseResult};
use num_traits::Zero;
use std::cmp::PartialEq;

/// Compressed Sparse Row (CSR) matrix
///
/// A sparse matrix format that compresses rows, making it efficient for
/// row operations and matrix-vector multiplication.
#[derive(Clone)]
pub struct CsrMatrix<T> {
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Row pointers (size rows+1)
    pub indptr: Vec<usize>,
    /// Column indices
    pub indices: Vec<usize>,
    /// Data values
    pub data: Vec<T>,
}

impl<T> CsrMatrix<T>
where
    T: Clone + Copy + Zero + PartialEq,
{
    /// Get the value at the specified position
    pub fn get(&self, row: usize, col: usize) -> T {
        // Check bounds
        if row >= self.rows || col >= self.cols {
            return T::zero();
        }

        // Find the element in the CSR format
        for j in self.indptr[row]..self.indptr[row + 1] {
            if self.indices[j] == col {
                return self.data[j];
            }
        }

        // Element not found, return zero
        T::zero()
    }

    /// Get the triplets (row indices, column indices, data)
    pub fn get_triplets(&self) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();

        for i in 0..self.rows {
            for j in self.indptr[i]..self.indptr[i + 1] {
                rows.push(i);
                cols.push(self.indices[j]);
                values.push(self.data[j]);
            }
        }

        (rows, cols, values)
    }
    /// Create a new CSR matrix from raw data
    ///
    /// # Arguments
    ///
    /// * `data` - Vector of non-zero values
    /// * `row_indices` - Vector of row indices for each non-zero value
    /// * `col_indices` - Vector of column indices for each non-zero value
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new CSR matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_sparse::csr::CsrMatrix;
    ///
    /// // Create a 3x3 sparse matrix with 5 non-zero elements
    /// let rows = vec![0, 0, 1, 2, 2];
    /// let cols = vec![0, 2, 2, 0, 1];
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let shape = (3, 3);
    ///
    /// let matrix = CsrMatrix::new(data.clone(), rows, cols, shape).unwrap();
    /// ```
    pub fn new(
        data: Vec<T>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        // Validate input data
        if data.len() != row_indices.len() || data.len() != col_indices.len() {
            return Err(SparseError::DimensionMismatch {
                expected: data.len(),
                found: std::cmp::min(row_indices.len(), col_indices.len()),
            });
        }

        let (rows, cols) = shape;

        // Check indices are within bounds
        if row_indices.iter().any(|&i| i >= rows) {
            return Err(SparseError::ValueError(
                "Row index out of bounds".to_string(),
            ));
        }

        if col_indices.iter().any(|&i| i >= cols) {
            return Err(SparseError::ValueError(
                "Column index out of bounds".to_string(),
            ));
        }

        // Convert triplet format to CSR
        // First, sort by row, then by column
        let mut triplets: Vec<(usize, usize, T)> = row_indices
            .into_iter()
            .zip(col_indices)
            .zip(data)
            .map(|((r, c), v)| (r, c, v))
            .collect();
        triplets.sort_by_key(|&(r, c, _)| (r, c));

        // Create indptr, indices, and data arrays
        let nnz = triplets.len();
        let mut indptr = vec![0; rows + 1];
        let mut indices = Vec::with_capacity(nnz);
        let mut data_out = Vec::with_capacity(nnz);

        // Count elements per row to build indptr
        for &(r, _, _) in &triplets {
            indptr[r + 1] += 1;
        }

        // Compute cumulative sum for indptr
        for i in 1..=rows {
            indptr[i] += indptr[i - 1];
        }

        // Fill indices and data
        for (_r, c, v) in triplets {
            indices.push(c);
            data_out.push(v);
        }

        Ok(CsrMatrix {
            rows,
            cols,
            indptr,
            indices,
            data: data_out,
        })
    }

    /// Create a new CSR matrix from raw CSR format
    ///
    /// # Arguments
    ///
    /// * `data` - Vector of non-zero values
    /// * `indptr` - Vector of row pointers (size rows+1)
    /// * `indices` - Vector of column indices
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new CSR matrix
    pub fn from_raw_csr(
        data: Vec<T>,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        let (rows, cols) = shape;

        // Validate input data
        if indptr.len() != rows + 1 {
            return Err(SparseError::DimensionMismatch {
                expected: rows + 1,
                found: indptr.len(),
            });
        }

        if data.len() != indices.len() {
            return Err(SparseError::DimensionMismatch {
                expected: data.len(),
                found: indices.len(),
            });
        }

        // Check if indptr is monotonically increasing
        for i in 1..indptr.len() {
            if indptr[i] < indptr[i - 1] {
                return Err(SparseError::ValueError(
                    "Row pointer array must be monotonically increasing".to_string(),
                ));
            }
        }

        // Check if the last indptr entry matches the data length
        if indptr[rows] != data.len() {
            return Err(SparseError::ValueError(
                "Last row pointer entry must match data length".to_string(),
            ));
        }

        // Check if indices are within bounds
        if indices.iter().any(|&i| i >= cols) {
            return Err(SparseError::ValueError(
                "Column index out of bounds".to_string(),
            ));
        }

        Ok(CsrMatrix {
            rows,
            cols,
            indptr,
            indices,
            data,
        })
    }

    /// Create a new empty CSR matrix
    ///
    /// # Arguments
    ///
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new empty CSR matrix
    pub fn empty(shape: (usize, usize)) -> Self {
        let (rows, cols) = shape;
        let indptr = vec![0; rows + 1];

        CsrMatrix {
            rows,
            cols,
            indptr,
            indices: Vec::new(),
            data: Vec::new(),
        }
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

    /// Get the number of non-zero elements in the matrix
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Convert to dense matrix (as Vec<Vec<T>>)
    pub fn to_dense(&self) -> Vec<Vec<T>>
    where
        T: Zero + Copy,
    {
        let mut result = vec![vec![T::zero(); self.cols]; self.rows];

        for (row_idx, row) in result.iter_mut().enumerate() {
            for j in self.indptr[row_idx]..self.indptr[row_idx + 1] {
                let col_idx = self.indices[j];
                row[col_idx] = self.data[j];
            }
        }

        result
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> Self {
        // Compute the number of non-zeros per column
        let mut col_counts = vec![0; self.cols];
        for &col in &self.indices {
            col_counts[col] += 1;
        }

        // Compute column pointers (cumulative sum)
        let mut col_ptrs = vec![0; self.cols + 1];
        for i in 0..self.cols {
            col_ptrs[i + 1] = col_ptrs[i] + col_counts[i];
        }

        // Fill the transposed matrix
        let nnz = self.nnz();
        let mut indices_t = vec![0; nnz];
        let mut data_t = vec![T::zero(); nnz];
        let mut col_counts = vec![0; self.cols];

        for row in 0..self.rows {
            for j in self.indptr[row]..self.indptr[row + 1] {
                let col = self.indices[j];
                let dest = col_ptrs[col] + col_counts[col];

                indices_t[dest] = row;
                data_t[dest] = self.data[j];
                col_counts[col] += 1;
            }
        }

        CsrMatrix {
            rows: self.cols,
            cols: self.rows,
            indptr: col_ptrs,
            indices: indices_t,
            data: data_t,
        }
    }
}

impl<
        T: Clone
            + Copy
            + std::ops::AddAssign
            + std::ops::MulAssign
            + std::cmp::PartialEq
            + std::fmt::Debug
            + num_traits::Zero
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>,
    > CsrMatrix<T>
{
    /// Check if matrix is symmetric
    ///
    /// # Returns
    ///
    /// * `true` if the matrix is symmetric, `false` otherwise
    pub fn is_symmetric(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }

        // Create a transposed matrix
        let transposed = self.transpose();

        // Compare the sparsity patterns and values
        if self.nnz() != transposed.nnz() {
            return false;
        }

        // Compare row by row
        for row in 0..self.rows {
            let self_start = self.indptr[row];
            let self_end = self.indptr[row + 1];
            let trans_start = transposed.indptr[row];
            let trans_end = transposed.indptr[row + 1];

            if self_end - self_start != trans_end - trans_start {
                return false;
            }

            // Create sorted columns and values for this row
            let mut self_entries: Vec<(usize, &T)> = (self_start..self_end)
                .map(|j| (self.indices[j], &self.data[j]))
                .collect();
            self_entries.sort_by_key(|(col, _)| *col);

            let mut trans_entries: Vec<(usize, &T)> = (trans_start..trans_end)
                .map(|j| (transposed.indices[j], &transposed.data[j]))
                .collect();
            trans_entries.sort_by_key(|(col, _)| *col);

            // Compare columns and values
            for i in 0..self_entries.len() {
                if self_entries[i].0 != trans_entries[i].0
                    || self_entries[i].1 != trans_entries[i].1
                {
                    return false;
                }
            }
        }

        true
    }

    /// Matrix-matrix multiplication
    ///
    /// # Arguments
    ///
    /// * `other` - Matrix to multiply with
    ///
    /// # Returns
    ///
    /// * Result containing the product matrix
    pub fn matmul(&self, other: &CsrMatrix<T>) -> SparseResult<CsrMatrix<T>> {
        if self.cols != other.rows {
            return Err(SparseError::DimensionMismatch {
                expected: self.cols,
                found: other.rows,
            });
        }

        // For simplicity, we'll implement this using dense operations
        // In a real implementation, you'd use a more efficient sparse algorithm
        let a_dense = self.to_dense();
        let b_dense = other.to_dense();

        let m = self.rows;
        let n = other.cols;
        let k = self.cols;

        let mut c_dense = vec![vec![T::zero(); n]; m];

        for (i, c_row) in c_dense.iter_mut().enumerate().take(m) {
            for j in 0..n {
                for l in 0..k {
                    let prod = a_dense[i][l] * b_dense[l][j];
                    c_row[j] += prod;
                }
            }
        }

        // Convert back to CSR format
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for (i, row) in c_dense.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                if *val != T::zero() {
                    row_indices.push(i);
                    col_indices.push(j);
                    values.push(*val);
                }
            }
        }

        CsrMatrix::new(values, row_indices, col_indices, (m, n))
    }

    /// Get row range for iterating over elements in a row
    ///
    /// # Arguments
    ///
    /// * `row` - Row index
    ///
    /// # Returns
    ///
    /// * Range of indices in the data and indices arrays for this row
    pub fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        assert!(row < self.rows, "Row index out of bounds");
        self.indptr[row]..self.indptr[row + 1]
    }

    /// Get column indices array
    pub fn col_indices(&self) -> &[usize] {
        &self.indices
    }
}

impl CsrMatrix<f64> {
    /// Matrix-vector multiplication
    ///
    /// # Arguments
    ///
    /// * `vec` - Vector to multiply with
    ///
    /// # Returns
    ///
    /// * Result of matrix-vector multiplication
    pub fn dot(&self, vec: &[f64]) -> SparseResult<Vec<f64>> {
        if vec.len() != self.cols {
            return Err(SparseError::DimensionMismatch {
                expected: self.cols,
                found: vec.len(),
            });
        }

        let mut result = vec![0.0; self.rows];

        for (row_idx, result_val) in result.iter_mut().enumerate() {
            for j in self.indptr[row_idx]..self.indptr[row_idx + 1] {
                let col_idx = self.indices[j];
                *result_val += self.data[j] * vec[col_idx];
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_csr_create() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();

        assert_eq!(matrix.shape(), (3, 3));
        assert_eq!(matrix.nnz(), 5);
    }

    #[test]
    fn test_csr_to_dense() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let dense = matrix.to_dense();

        let expected = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 0.0, 3.0],
            vec![4.0, 5.0, 0.0],
        ];

        assert_eq!(dense, expected);
    }

    #[test]
    fn test_csr_dot() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();

        // Matrix:
        // [1 0 2]
        // [0 0 3]
        // [4 5 0]

        let vec = vec![1.0, 2.0, 3.0];
        let result = matrix.dot(&vec).unwrap();

        // Expected:
        // 1*1 + 0*2 + 2*3 = 7
        // 0*1 + 0*2 + 3*3 = 9
        // 4*1 + 5*2 + 0*3 = 14
        let expected = vec![7.0, 9.0, 14.0];

        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_csr_transpose() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
        let transposed = matrix.transpose();

        assert_eq!(transposed.shape(), (3, 3));
        assert_eq!(transposed.nnz(), 5);

        let dense = transposed.to_dense();
        let expected = vec![
            vec![1.0, 0.0, 4.0],
            vec![0.0, 0.0, 5.0],
            vec![2.0, 3.0, 0.0],
        ];

        assert_eq!(dense, expected);
    }
}
