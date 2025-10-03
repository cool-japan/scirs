//! Diagonal (DIA) matrix format
//!
//! This module provides the DIA matrix format implementation, which is
//! efficient for matrices with values concentrated on a small number of diagonals.

use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::Zero;

/// Diagonal (DIA) matrix
///
/// A sparse matrix format that stores diagonals, making it efficient for
/// matrices with values concentrated on a small number of diagonals.
pub struct DiaMatrix<T> {
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Diagonals data (n_diags x max(rows, cols))
    data: Vec<Vec<T>>,
    /// Diagonal offsets from the main diagonal
    offsets: Vec<isize>,
}

impl<T> DiaMatrix<T>
where
    T: Clone + Copy + Zero + std::cmp::PartialEq,
{
    /// Create a new DIA matrix from raw data
    ///
    /// # Arguments
    ///
    /// * `data` - Diagonals data (n_diags x max(rows, cols))
    /// * `offsets` - Diagonal offsets from the main diagonal
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new DIA matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_sparse::dia::DiaMatrix;
    ///
    /// // Create a 3x3 sparse matrix with main diagonal and upper diagonal
    /// let data = vec![
    ///     vec![1.0, 2.0, 3.0], // Main diagonal
    ///     vec![4.0, 5.0, 0.0], // Upper diagonal (k=1)
    /// ];
    /// let offsets = vec![0, 1]; // Main diagonal and k=1
    /// let shape = (3, 3);
    ///
    /// let matrix = DiaMatrix::new(data, offsets, shape).unwrap();
    /// ```
    pub fn new(
        data: Vec<Vec<T>>,
        offsets: Vec<isize>,
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        let (rows, cols) = shape;
        let max_dim = rows.max(cols);

        // Validate input data
        if data.len() != offsets.len() {
            return Err(SparseError::DimensionMismatch {
                expected: data.len(),
                found: offsets.len(),
            });
        }

        for diag in data.iter() {
            if diag.len() != max_dim {
                return Err(SparseError::DimensionMismatch {
                    expected: max_dim,
                    found: diag.len(),
                });
            }
        }

        Ok(DiaMatrix {
            rows,
            cols,
            data,
            offsets,
        })
    }

    /// Create a new empty DIA matrix
    ///
    /// # Arguments
    ///
    /// * `shape` - Tuple containing the matrix dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new empty DIA matrix
    pub fn empty(shape: (usize, usize)) -> Self {
        let (rows, cols) = shape;

        DiaMatrix {
            rows,
            cols,
            data: Vec::new(),
            offsets: Vec::new(),
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
        let mut count = 0;

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.data[diag_idx];

            // Calculate valid range for this diagonal
            let mut start = 0;
            let mut end = self.rows.min(self.cols);

            if offset < 0 {
                start = (-offset) as usize;
            }

            if offset > 0 {
                end = (self.rows as isize - offset) as usize;
            }

            // Count non-zeros in the valid range
            for val in diag.iter().skip(start).take(end - start) {
                if *val != T::zero() {
                    count += 1;
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

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.data[diag_idx];

            if offset >= 0 {
                // Upper diagonal
                let offset = offset as usize;
                for i in 0..self.rows.min(self.cols.saturating_sub(offset)) {
                    result[i][i + offset] = diag[i];
                }
            } else {
                // Lower diagonal
                let offset = (-offset) as usize;
                for i in 0..self.cols.min(self.rows.saturating_sub(offset)) {
                    result[i + offset][i] = diag[i];
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
    fn test_dia_create() {
        // Create a 3x3 sparse matrix with main diagonal and upper diagonal
        let data = vec![
            vec![1.0, 2.0, 3.0], // Main diagonal
            vec![4.0, 5.0, 0.0], // Upper diagonal (k=1)
        ];
        let offsets = vec![0, 1]; // Main diagonal and k=1
        let shape = (3, 3);

        let matrix = DiaMatrix::new(data, offsets, shape).unwrap();

        assert_eq!(matrix.shape(), (3, 3));
        assert_eq!(matrix.nnz(), 5); // 3 on main diagonal, 2 on upper diagonal
    }

    #[test]
    fn test_dia_to_dense() {
        // Create a 3x3 sparse matrix with main diagonal and upper diagonal
        let data = vec![
            vec![1.0, 2.0, 3.0], // Main diagonal
            vec![4.0, 5.0, 0.0], // Upper diagonal (k=1)
        ];
        let offsets = vec![0, 1]; // Main diagonal and k=1
        let shape = (3, 3);

        let matrix = DiaMatrix::new(data, offsets, shape).unwrap();
        let dense = matrix.to_dense();

        let expected = vec![
            vec![1.0, 4.0, 0.0],
            vec![0.0, 2.0, 5.0],
            vec![0.0, 0.0, 3.0],
        ];

        assert_eq!(dense, expected);
    }
}
