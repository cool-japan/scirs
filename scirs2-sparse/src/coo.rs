//! Coordinate (COO) matrix format
//!
//! This module provides the COO matrix format implementation, which is
//! efficient for incremental matrix construction.

use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::Zero;
use std::cmp::PartialEq;

/// Coordinate (COO) matrix
///
/// A sparse matrix format that stores triplets (row, column, value),
/// making it efficient for construction and modification.
pub struct CooMatrix<T> {
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Row indices
    row_indices: Vec<usize>,
    /// Column indices
    col_indices: Vec<usize>,
    /// Data values
    data: Vec<T>,
}

impl<T> CooMatrix<T>
where
    T: Clone + Copy + Zero + PartialEq,
{
    /// Get the triplets (row indices, column indices, data)
    pub fn get_triplets(&self) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        (
            self.row_indices.clone(),
            self.col_indices.clone(),
            self.data.clone(),
        )
    }
    /// Create a new COO matrix from raw data
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
    /// * A new COO matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_sparse::coo::CooMatrix;
    ///
    /// // Create a 3x3 sparse matrix with 5 non-zero elements
    /// let rows = vec![0, 0, 1, 2, 2];
    /// let cols = vec![0, 2, 2, 0, 1];
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let shape = (3, 3);
    ///
    /// let matrix = CooMatrix::new(data, rows, cols, shape).unwrap();
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

        // Check _indices are within bounds
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

        Ok(CooMatrix {
            rows,
            cols,
            row_indices,
            col_indices,
            data,
        })
    }

    /// Create a new empty COO matrix
    ///
    /// # Arguments
    ///
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new empty COO matrix
    pub fn empty(shape: (usize, usize)) -> Self {
        let (rows, cols) = shape;

        CooMatrix {
            rows,
            cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Add a value to the matrix at the specified position
    ///
    /// # Arguments
    ///
    /// * `row` - Row index
    /// * `col` - Column index
    /// * `value` - Value to add
    ///
    /// # Returns
    ///
    /// * Ok(()) if successful, Error otherwise
    pub fn add_element(&mut self, row: usize, col: usize, value: T) -> SparseResult<()> {
        if row >= self.rows || col >= self.cols {
            return Err(SparseError::ValueError(
                "Row or column index out of bounds".to_string(),
            ));
        }

        self.row_indices.push(row);
        self.col_indices.push(col);
        self.data.push(value);

        Ok(())
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

    /// Get the row indices array
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Get the column indices array
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Get the data array
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Convert to dense matrix (as Vec<Vec<T>>)
    pub fn to_dense(&self) -> Vec<Vec<T>>
    where
        T: Zero + Copy,
    {
        let mut result = vec![vec![T::zero(); self.cols]; self.rows];

        for i in 0..self.data.len() {
            let row = self.row_indices[i];
            let col = self.col_indices[i];
            result[row][col] = self.data[i];
        }

        result
    }

    /// Convert to CSR format
    pub fn to_csr(&self) -> crate::csr::CsrMatrix<T> {
        crate::csr::CsrMatrix::new(
            self.data.clone(),
            self.row_indices.clone(),
            self.col_indices.clone(),
            (self.rows, self.cols),
        )
        .unwrap()
    }

    /// Convert to CSC format
    pub fn to_csc(&self) -> crate::csc::CscMatrix<T> {
        crate::csc::CscMatrix::new(
            self.data.clone(),
            self.row_indices.clone(),
            self.col_indices.clone(),
            (self.rows, self.cols),
        )
        .unwrap()
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> Self {
        let mut transposed_data = Vec::with_capacity(self.data.len());
        let mut transposed_row_indices = Vec::with_capacity(self.row_indices.len());
        let mut transposed_col_indices = Vec::with_capacity(self.col_indices.len());

        for i in 0..self.data.len() {
            transposed_data.push(self.data[i]);
            transposed_row_indices.push(self.col_indices[i]);
            transposed_col_indices.push(self.row_indices[i]);
        }

        CooMatrix {
            rows: self.cols,
            cols: self.rows,
            row_indices: transposed_row_indices,
            col_indices: transposed_col_indices,
            data: transposed_data,
        }
    }

    /// Sort the matrix elements by row, then column
    pub fn sort_by_row_col(&mut self) {
        let mut indices: Vec<usize> = (0..self.data.len()).collect();
        indices.sort_by_key(|&i| (self.row_indices[i], self.col_indices[i]));

        let row_indices = self.row_indices.clone();
        let col_indices = self.col_indices.clone();
        let data = self.data.clone();

        for (i, &idx) in indices.iter().enumerate() {
            self.row_indices[i] = row_indices[idx];
            self.col_indices[i] = col_indices[idx];
            self.data[i] = data[idx];
        }
    }

    /// Sort the matrix elements by column, then row
    pub fn sort_by_col_row(&mut self) {
        let mut indices: Vec<usize> = (0..self.data.len()).collect();
        indices.sort_by_key(|&i| (self.col_indices[i], self.row_indices[i]));

        let row_indices = self.row_indices.clone();
        let col_indices = self.col_indices.clone();
        let data = self.data.clone();

        for (i, &idx) in indices.iter().enumerate() {
            self.row_indices[i] = row_indices[idx];
            self.col_indices[i] = col_indices[idx];
            self.data[i] = data[idx];
        }
    }

    /// Get the value at the specified position
    pub fn get(&self, row: usize, col: usize) -> T
    where
        T: Zero,
    {
        for i in 0..self.data.len() {
            if self.row_indices[i] == row && self.col_indices[i] == col {
                return self.data[i];
            }
        }
        T::zero()
    }

    /// Sum duplicate entries (elements with the same row and column indices)
    pub fn sum_duplicates(&mut self)
    where
        T: std::ops::Add<Output = T>,
    {
        if self.data.is_empty() {
            return;
        }

        // Sort by row and column
        self.sort_by_row_col();

        let mut unique_row_indices = Vec::new();
        let mut unique_col_indices = Vec::new();
        let mut unique_data = Vec::new();

        let mut current_row = self.row_indices[0];
        let mut current_col = self.col_indices[0];
        let mut current_val = self.data[0];

        for i in 1..self.data.len() {
            if self.row_indices[i] == current_row && self.col_indices[i] == current_col {
                // Same position, add values
                current_val = current_val + self.data[i];
            } else {
                // New position, store the previous one
                unique_row_indices.push(current_row);
                unique_col_indices.push(current_col);
                unique_data.push(current_val);

                // Update current position
                current_row = self.row_indices[i];
                current_col = self.col_indices[i];
                current_val = self.data[i];
            }
        }

        // Add the last element
        unique_row_indices.push(current_row);
        unique_col_indices.push(current_col);
        unique_data.push(current_val);

        // Update the matrix
        self.row_indices = unique_row_indices;
        self.col_indices = unique_col_indices;
        self.data = unique_data;
    }
}

impl CooMatrix<f64> {
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

        for i in 0..self.data.len() {
            let row = self.row_indices[i];
            let col = self.col_indices[i];
            result[row] += self.data[i] * vec[col];
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_coo_create() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CooMatrix::new(data, rows, cols, shape).unwrap();

        assert_eq!(matrix.shape(), (3, 3));
        assert_eq!(matrix.nnz(), 5);
    }

    #[test]
    fn test_coo_add_element() {
        // Create an empty matrix
        let mut matrix = CooMatrix::<f64>::empty((3, 3));

        // Add elements
        matrix.add_element(0, 0, 1.0).unwrap();
        matrix.add_element(0, 2, 2.0).unwrap();
        matrix.add_element(1, 2, 3.0).unwrap();
        matrix.add_element(2, 0, 4.0).unwrap();
        matrix.add_element(2, 1, 5.0).unwrap();

        assert_eq!(matrix.nnz(), 5);

        // Adding element out of bounds should fail
        assert!(matrix.add_element(3, 0, 6.0).is_err());
        assert!(matrix.add_element(0, 3, 6.0).is_err());
    }

    #[test]
    fn test_coo_to_dense() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CooMatrix::new(data, rows, cols, shape).unwrap();
        let dense = matrix.to_dense();

        let expected = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 0.0, 3.0],
            vec![4.0, 5.0, 0.0],
        ];

        assert_eq!(dense, expected);
    }

    #[test]
    fn test_coo_dot() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CooMatrix::new(data, rows, cols, shape).unwrap();

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
        let expected = [7.0, 9.0, 14.0];

        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_coo_transpose() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let matrix = CooMatrix::new(data, rows, cols, shape).unwrap();
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

    #[test]
    fn test_coo_sort_and_sum_duplicates() {
        // Create a matrix with duplicate entries
        let rows = vec![0, 0, 0, 1, 1, 2];
        let cols = vec![0, 0, 1, 0, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = (3, 2);

        let mut matrix = CooMatrix::new(data, rows, cols, shape).unwrap();
        matrix.sum_duplicates();

        assert_eq!(matrix.nnz(), 4); // Should have 4 unique entries after summing

        let dense = matrix.to_dense();
        let expected = vec![vec![3.0, 3.0], vec![9.0, 0.0], vec![0.0, 6.0]];

        assert_eq!(dense, expected);
    }

    #[test]
    fn test_coo_to_csr_to_csc() {
        // Create a 3x3 sparse matrix with 5 non-zero elements
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let coo_matrix = CooMatrix::new(data, rows, cols, shape).unwrap();

        // Convert to CSR and CSC
        let csr_matrix = coo_matrix.to_csr();
        let csc_matrix = coo_matrix.to_csc();

        // Convert back to dense and compare
        let dense_from_coo = coo_matrix.to_dense();
        let dense_from_csr = csr_matrix.to_dense();
        let dense_from_csc = csc_matrix.to_dense();

        assert_eq!(dense_from_coo, dense_from_csr);
        assert_eq!(dense_from_coo, dense_from_csc);
    }
}
