//! List of Lists (LIL) matrix format
//!
//! This module provides the LIL matrix format implementation, which is
//! efficient for row-based operations and incremental matrix construction.

use scirs2_core::numeric::Zero;

/// List of Lists (LIL) matrix
///
/// A sparse matrix format that stores data as a list of lists,
/// making it efficient for row-based operations and incremental construction.
pub struct LilMatrix<T> {
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Data for each row (list of lists of values)
    data: Vec<Vec<T>>,
    /// Column indices for each row
    indices: Vec<Vec<usize>>,
}

impl<T> LilMatrix<T>
where
    T: Clone + Copy + Zero + std::cmp::PartialEq,
{
    /// Create a new LIL matrix
    ///
    /// # Arguments
    ///
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new empty LIL matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_sparse::lil::LilMatrix;
    ///
    /// // Create a 3x3 sparse matrix
    /// let mut matrix = LilMatrix::<f64>::new((3, 3));
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

        let data = vec![Vec::new(); rows];
        let indices = vec![Vec::new(); rows];

        LilMatrix {
            rows,
            cols,
            data,
            indices,
        }
    }

    /// Set a value in the matrix
    ///
    /// # Arguments
    ///
    /// * `row` - Row index
    /// * `col` - Column index
    /// * `value` - Value to set
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        if row >= self.rows || col >= self.cols {
            return;
        }

        // Find existing entry or insertion point
        match self.indices[row].binary_search(&col) {
            Ok(idx) => {
                // Column already exists
                if value == T::zero() {
                    // Remove zero entry
                    self.data[row].remove(idx);
                    self.indices[row].remove(idx);
                } else {
                    // Update non-zero value
                    self.data[row][idx] = value;
                }
            }
            Err(idx) => {
                // Column doesn't exist
                if value != T::zero() {
                    // Insert non-zero value
                    self.data[row].insert(idx, value);
                    self.indices[row].insert(idx, col);
                }
            }
        }
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

        match self.indices[row].binary_search(&col) {
            Ok(idx) => self.data[row][idx],
            Err(_) => T::zero(),
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
        self.indices.iter().map(|row| row.len()).sum()
    }

    /// Convert to dense matrix (as Vec<Vec<T>>)
    pub fn to_dense(&self) -> Vec<Vec<T>>
    where
        T: Zero + Copy,
    {
        let mut result = vec![vec![T::zero(); self.cols]; self.rows];

        for (row, (row_indices, row_data)) in self
            .indices
            .iter()
            .zip(&self.data)
            .enumerate()
            .take(self.rows)
        {
            for (idx, &col) in row_indices.iter().enumerate() {
                result[row][col] = row_data[idx];
            }
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

        for row in 0..self.rows {
            for (idx, &col) in self.indices[row].iter().enumerate() {
                data.push(self.data[row][idx]);
                row_indices.push(row);
                col_indices.push(col);
            }
        }

        (data, row_indices, col_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lil_create_and_access() {
        // Create a 3x3 sparse matrix
        let mut matrix = LilMatrix::<f64>::new((3, 3));

        // Set some values
        matrix.set(0, 0, 1.0);
        matrix.set(0, 2, 2.0);
        matrix.set(1, 2, 3.0);
        matrix.set(2, 0, 4.0);
        matrix.set(2, 1, 5.0);

        assert_eq!(matrix.nnz(), 5);

        // Access values
        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(0, 1), 0.0); // Zero entry
        assert_eq!(matrix.get(0, 2), 2.0);
        assert_eq!(matrix.get(1, 2), 3.0);
        assert_eq!(matrix.get(2, 0), 4.0);
        assert_eq!(matrix.get(2, 1), 5.0);

        // Set a value to zero should remove it
        matrix.set(0, 0, 0.0);
        assert_eq!(matrix.nnz(), 4);
        assert_eq!(matrix.get(0, 0), 0.0);

        // Out of bounds access should return zero
        assert_eq!(matrix.get(3, 0), 0.0);
        assert_eq!(matrix.get(0, 3), 0.0);
    }

    #[test]
    fn test_lil_to_dense() {
        // Create a 3x3 sparse matrix
        let mut matrix = LilMatrix::<f64>::new((3, 3));

        // Set some values
        matrix.set(0, 0, 1.0);
        matrix.set(0, 2, 2.0);
        matrix.set(1, 2, 3.0);
        matrix.set(2, 0, 4.0);
        matrix.set(2, 1, 5.0);

        let dense = matrix.to_dense();

        let expected = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 0.0, 3.0],
            vec![4.0, 5.0, 0.0],
        ];

        assert_eq!(dense, expected);
    }

    #[test]
    fn test_lil_to_coo() {
        // Create a 3x3 sparse matrix
        let mut matrix = LilMatrix::<f64>::new((3, 3));

        // Set some values
        matrix.set(0, 0, 1.0);
        matrix.set(0, 2, 2.0);
        matrix.set(1, 2, 3.0);
        matrix.set(2, 0, 4.0);
        matrix.set(2, 1, 5.0);

        let (data, row_indices, col_indices) = matrix.to_coo();

        // Check that all entries are present
        assert_eq!(data.len(), 5);
        assert_eq!(row_indices.len(), 5);
        assert_eq!(col_indices.len(), 5);

        // Validate presence of all elements
        for i in 0..data.len() {
            let row = row_indices[i];
            let col = col_indices[i];
            let val = data[i];

            assert_eq!(matrix.get(row, col), val);
        }
    }
}
