//! Dictionary of Keys (DOK) matrix format
//!
//! This module provides the DOK matrix format implementation, which is
//! efficient for incremental matrix construction.

use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::Zero;
use std::collections::HashMap;

/// Dictionary of Keys (DOK) matrix
///
/// A sparse matrix format that stores elements in a dictionary (hash map),
/// making it efficient for incremental construction.
pub struct DokMatrix<T> {
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Dictionary of (row, col) -> value
    data: HashMap<(usize, usize), T>,
}

impl<T> DokMatrix<T>
where
    T: Clone + Copy + Zero + std::cmp::PartialEq,
{
    /// Create a new DOK matrix
    ///
    /// # Arguments
    ///
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new empty DOK matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_sparse::dok::DokMatrix;
    ///
    /// // Create a 3x3 sparse matrix
    /// let mut matrix = DokMatrix::<f64>::new((3, 3));
    ///
    /// // Set some values
    /// matrix.set(0, 0, 1.0);
    /// matrix.set(0, 2, 2.0);
    /// matrix.set(1, 2, 3.0);
    /// matrix.set(2, 0, 4.0);
    /// matrix.set(2, 1, 5.0);
    /// ```
    pub fn new(shape: (usize, usize)) -> Self {
        let (rows, cols) = shape;

        DokMatrix {
            rows,
            cols,
            data: HashMap::new(),
        }
    }

    /// Set a value in the matrix
    ///
    /// # Arguments
    ///
    /// * `row` - Row index
    /// * `col` - Column index
    /// * `value` - Value to set
    ///
    /// # Returns
    ///
    /// * Ok(()) if successful, Error otherwise
    pub fn set(&mut self, row: usize, col: usize, value: T) -> SparseResult<()> {
        if row >= self.rows || col >= self.cols {
            return Err(SparseError::ValueError(
                "Row or column index out of bounds".to_string(),
            ));
        }

        if value == T::zero() {
            // Remove zero entries
            self.data.remove(&(row, col));
        } else {
            // Set non-zero value
            self.data.insert((row, col), value);
        }

        Ok(())
    }

    /// Get a value from the matrix
    ///
    /// # Arguments
    ///
    /// * `row` - Row index
    /// * `col` - Column index
    ///
    /// # Returns
    ///
    /// * Value at the specified position, or zero if not set
    pub fn get(&self, row: usize, col: usize) -> T {
        if row >= self.rows || col >= self.cols {
            return T::zero();
        }

        *self.data.get(&(row, col)).unwrap_or(&T::zero())
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

        for (&(row, col), &value) in &self.data {
            result[row][col] = value;
        }

        result
    }

    /// Convert to COO representation
    ///
    /// # Returns
    ///
    /// * Tuple of (data, row_indices, col_indices)
    pub fn to_coo(&self) -> (Vec<T>, Vec<usize>, Vec<usize>) {
        let nnz = self.nnz();
        let mut data = Vec::with_capacity(nnz);
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);

        // Sort by row, then column for deterministic output
        let mut entries: Vec<_> = self.data.iter().collect();
        entries.sort_by_key(|(&(row, col), _)| (row, col));

        for (&(row, col), &value) in entries {
            data.push(value);
            row_indices.push(row);
            col_indices.push(col);
        }

        (data, row_indices, col_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dok_create_and_access() {
        // Create a 3x3 sparse matrix
        let mut matrix = DokMatrix::<f64>::new((3, 3));

        // Set some values
        matrix.set(0, 0, 1.0).unwrap();
        matrix.set(0, 2, 2.0).unwrap();
        matrix.set(1, 2, 3.0).unwrap();
        matrix.set(2, 0, 4.0).unwrap();
        matrix.set(2, 1, 5.0).unwrap();

        assert_eq!(matrix.nnz(), 5);

        // Access values
        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(0, 1), 0.0); // Zero entry
        assert_eq!(matrix.get(0, 2), 2.0);
        assert_eq!(matrix.get(1, 2), 3.0);
        assert_eq!(matrix.get(2, 0), 4.0);
        assert_eq!(matrix.get(2, 1), 5.0);

        // Set a value to zero should remove it
        matrix.set(0, 0, 0.0).unwrap();
        assert_eq!(matrix.nnz(), 4);
        assert_eq!(matrix.get(0, 0), 0.0);

        // Out of bounds access should return zero
        assert_eq!(matrix.get(3, 0), 0.0);
        assert_eq!(matrix.get(0, 3), 0.0);
    }

    #[test]
    fn test_dok_to_dense() {
        // Create a 3x3 sparse matrix
        let mut matrix = DokMatrix::<f64>::new((3, 3));

        // Set some values
        matrix.set(0, 0, 1.0).unwrap();
        matrix.set(0, 2, 2.0).unwrap();
        matrix.set(1, 2, 3.0).unwrap();
        matrix.set(2, 0, 4.0).unwrap();
        matrix.set(2, 1, 5.0).unwrap();

        let dense = matrix.to_dense();

        let expected = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 0.0, 3.0],
            vec![4.0, 5.0, 0.0],
        ];

        assert_eq!(dense, expected);
    }

    #[test]
    fn test_dok_to_coo() {
        // Create a 3x3 sparse matrix
        let mut matrix = DokMatrix::<f64>::new((3, 3));

        // Set some values
        matrix.set(0, 0, 1.0).unwrap();
        matrix.set(0, 2, 2.0).unwrap();
        matrix.set(1, 2, 3.0).unwrap();
        matrix.set(2, 0, 4.0).unwrap();
        matrix.set(2, 1, 5.0).unwrap();

        let (data, row_indices, col_indices) = matrix.to_coo();

        // Check that all entries are present
        assert_eq!(data.len(), 5);
        assert_eq!(row_indices.len(), 5);
        assert_eq!(col_indices.len(), 5);

        // Check the content (sorted by row, then column)
        let expected_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected_rows = vec![0, 0, 1, 2, 2];
        let expected_cols = vec![0, 2, 2, 0, 1];

        assert_eq!(data, expected_data);
        assert_eq!(row_indices, expected_rows);
        assert_eq!(col_indices, expected_cols);
    }
}
