//! Symmetric Compressed Sparse Row (SymCSR) module
//!
//! This module provides a specialized implementation of the CSR format
//! optimized for symmetric matrices, storing only the lower or upper
//! triangular part of the matrix.

use crate::csr::CsrMatrix;
use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use num_traits::Float;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

/// Symmetric Compressed Sparse Row (SymCSR) matrix
///
/// This format stores only the lower triangular part of a symmetric matrix
/// to save memory and improve performance. Operations are optimized to
/// take advantage of symmetry when possible.
///
/// # Note
///
/// All operations maintain symmetry implicitly.
#[derive(Debug, Clone)]
pub struct SymCsrMatrix<T>
where
    T: Float + Debug + Copy,
{
    /// CSR format data for the lower triangular part (including diagonal)
    pub data: Vec<T>,

    /// Row pointers (indptr): indices where each row starts in indices array
    pub indptr: Vec<usize>,

    /// Column indices for each non-zero element
    pub indices: Vec<usize>,

    /// Matrix shape (rows, cols), always square
    pub shape: (usize, usize),
}

impl<T> SymCsrMatrix<T>
where
    T: Float + Debug + Copy,
{
    /// Create a new symmetric CSR matrix from raw data
    ///
    /// # Arguments
    ///
    /// * `data` - Non-zero values in the lower triangular part
    /// * `indptr` - Row pointers
    /// * `indices` - Column indices
    /// * `shape` - Matrix shape (n, n)
    ///
    /// # Returns
    ///
    /// A symmetric CSR matrix
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The shape is not square
    /// - The indices array is incompatible with indptr
    /// - Any column index is out of bounds
    pub fn new(
        data: Vec<T>,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        let (rows, cols) = shape;

        // Ensure matrix is square
        if rows != cols {
            return Err(SparseError::ValueError(
                "Symmetric matrix must be square".to_string(),
            ));
        }

        // Check indptr length
        if indptr.len() != rows + 1 {
            return Err(SparseError::ValueError(format!(
                "indptr length ({}) must be equal to rows + 1 ({})",
                indptr.len(),
                rows + 1
            )));
        }

        // Check data and indices lengths
        let nnz = indices.len();
        if data.len() != nnz {
            return Err(SparseError::ValueError(format!(
                "data length ({}) must match indices length ({})",
                data.len(),
                nnz
            )));
        }

        // Check last indptr value
        if let Some(&last) = indptr.last() {
            if last != nnz {
                return Err(SparseError::ValueError(format!(
                    "Last indptr value ({}) must equal nnz ({})",
                    last, nnz
                )));
            }
        }

        // Check that row and column indices are within bounds
        for (i, &row_start) in indptr.iter().enumerate().take(rows) {
            let row_end = indptr[i + 1];

            for &col in &indices[row_start..row_end] {
                if col >= cols {
                    return Err(SparseError::IndexOutOfBounds {
                        index: (i, col),
                        shape: (rows, cols),
                    });
                }

                // For symmetric matrix, ensure we only store the lower triangular part
                if col > i {
                    return Err(SparseError::ValueError(
                        "Symmetric CSR should only store the lower triangular part".to_string(),
                    ));
                }
            }
        }

        Ok(Self {
            data,
            indptr,
            indices,
            shape,
        })
    }

    /// Convert a regular CSR matrix to symmetric CSR format
    ///
    /// This will verify that the matrix is symmetric and extract
    /// the lower triangular part.
    ///
    /// # Arguments
    ///
    /// * `matrix` - CSR matrix to convert
    ///
    /// # Returns
    ///
    /// A symmetric CSR matrix
    pub fn from_csr(matrix: &CsrMatrix<T>) -> SparseResult<Self> {
        let (rows, cols) = matrix.shape();

        // Ensure matrix is square
        if rows != cols {
            return Err(SparseError::ValueError(
                "Symmetric matrix must be square".to_string(),
            ));
        }

        // Check if the matrix is symmetric
        if !Self::is_symmetric(matrix) {
            return Err(SparseError::ValueError(
                "Matrix must be symmetric to convert to SymCSR format".to_string(),
            ));
        }

        // Extract the lower triangular part
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for i in 0..rows {
            for j in matrix.indptr[i]..matrix.indptr[i + 1] {
                let col = matrix.indices[j];

                // Only include elements in lower triangular part (including diagonal)
                if col <= i {
                    data.push(matrix.data[j]);
                    indices.push(col);
                }
            }

            indptr.push(data.len());
        }

        Ok(Self {
            data,
            indptr,
            indices,
            shape: (rows, cols),
        })
    }

    /// Check if a CSR matrix is symmetric
    ///
    /// # Arguments
    ///
    /// * `matrix` - CSR matrix to check
    ///
    /// # Returns
    ///
    /// `true` if the matrix is symmetric, `false` otherwise
    pub fn is_symmetric(matrix: &CsrMatrix<T>) -> bool {
        let (rows, cols) = matrix.shape();

        // Must be square
        if rows != cols {
            return false;
        }

        // Compare each element (i,j) with (j,i)
        for i in 0..rows {
            for j_ptr in matrix.indptr[i]..matrix.indptr[i + 1] {
                let j = matrix.indices[j_ptr];
                let val = matrix.data[j_ptr];

                // Find the corresponding (j,i) element
                let i_val = matrix.get(j, i);

                // Check if a[i,j] == a[j,i] with sufficient tolerance
                let diff = (val - i_val).abs();
                let epsilon = T::epsilon() * T::from(100.0).unwrap();
                if diff > epsilon {
                    return false;
                }
            }
        }

        true
    }

    /// Get the shape of the matrix
    ///
    /// # Returns
    ///
    /// A tuple (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get the number of stored non-zero elements
    ///
    /// # Returns
    ///
    /// The number of non-zero elements in the lower triangular part
    pub fn nnz_stored(&self) -> usize {
        self.data.len()
    }

    /// Get the total number of non-zero elements in the full matrix
    ///
    /// # Returns
    ///
    /// The total number of non-zero elements in the full symmetric matrix
    pub fn nnz(&self) -> usize {
        let diag_count = (0..self.shape.0)
            .filter(|&i| {
                // Count diagonal elements that are non-zero
                let row_start = self.indptr[i];
                let row_end = self.indptr[i + 1];
                (row_start..row_end).any(|j_ptr| self.indices[j_ptr] == i)
            })
            .count();

        let offdiag_count = self.data.len() - diag_count;

        // Diagonal elements count once, off-diagonal elements count twice
        diag_count + 2 * offdiag_count
    }

    /// Get a single element from the matrix
    ///
    /// # Arguments
    ///
    /// * `row` - Row index
    /// * `col` - Column index
    ///
    /// # Returns
    ///
    /// The value at position (row, col)
    pub fn get(&self, row: usize, col: usize) -> T {
        // Check bounds
        if row >= self.shape.0 || col >= self.shape.1 {
            return T::zero();
        }

        // For symmetric matrix, if (row,col) is in upper triangular part,
        // we look for (col,row) in the lower triangular part
        let (actual_row, actual_col) = if row < col { (col, row) } else { (row, col) };

        // Search for the element
        for j in self.indptr[actual_row]..self.indptr[actual_row + 1] {
            if self.indices[j] == actual_col {
                return self.data[j];
            }
        }

        T::zero()
    }

    /// Convert to standard CSR matrix (reconstructing full symmetric matrix)
    ///
    /// # Returns
    ///
    /// A standard CSR matrix with both upper and lower triangular parts
    pub fn to_csr(&self) -> SparseResult<CsrMatrix<T>> {
        let n = self.shape.0;

        // First, convert to triplet format for the full symmetric matrix
        let mut data = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        for i in 0..n {
            // Add elements from lower triangular part (directly stored)
            for j_ptr in self.indptr[i]..self.indptr[i + 1] {
                let j = self.indices[j_ptr];
                let val = self.data[j_ptr];

                // Add the element itself
                row_indices.push(i);
                col_indices.push(j);
                data.push(val);

                // Add its symmetric counterpart (if not on diagonal)
                if i != j {
                    row_indices.push(j);
                    col_indices.push(i);
                    data.push(val);
                }
            }
        }

        // Create the CSR matrix from triplets
        CsrMatrix::new(data, row_indices, col_indices, self.shape)
    }

    /// Convert to dense matrix
    ///
    /// # Returns
    ///
    /// A dense matrix representation as a vector of vectors
    pub fn to_dense(&self) -> Vec<Vec<T>> {
        let n = self.shape.0;
        let mut dense = vec![vec![T::zero(); n]; n];

        // Fill the lower triangular part (directly from stored data)
        for (i, row) in dense.iter_mut().enumerate().take(n) {
            for j_ptr in self.indptr[i]..self.indptr[i + 1] {
                let j = self.indices[j_ptr];
                row[j] = self.data[j_ptr];
            }
        }

        // Fill the upper triangular part (from symmetry)
        for i in 0..n {
            for j in 0..i {
                dense[j][i] = dense[i][j];
            }
        }

        dense
    }
}

/// Array-based SymCSR implementation compatible with SparseArray trait
#[derive(Debug, Clone)]
pub struct SymCsrArray<T>
where
    T: Float + Debug + Copy,
{
    /// Inner matrix
    inner: SymCsrMatrix<T>,
}

impl<T> SymCsrArray<T>
where
    T: Float
        + Debug
        + Copy
        + 'static
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    /// Create a new SymCSR array from a SymCSR matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - Symmetric CSR matrix
    ///
    /// # Returns
    ///
    /// SymCSR array
    pub fn new(matrix: SymCsrMatrix<T>) -> Self {
        Self { inner: matrix }
    }

    /// Create a SymCSR array from a regular CSR array
    ///
    /// # Arguments
    ///
    /// * `array` - CSR array to convert
    ///
    /// # Returns
    ///
    /// A symmetric CSR array
    pub fn from_csr_array(array: &CsrArray<T>) -> SparseResult<Self> {
        let shape = array.shape();
        let (rows, cols) = shape;

        // Ensure matrix is square
        if rows != cols {
            return Err(SparseError::ValueError(
                "Symmetric matrix must be square".to_string(),
            ));
        }

        // Create a temporary CSR matrix to check symmetry
        let csr_matrix = CsrMatrix::new(
            array.get_data().to_vec(),
            array.get_indptr().to_vec(),
            array.get_indices().to_vec(),
            shape,
        )?;

        // Convert to symmetric CSR
        let sym_csr = SymCsrMatrix::from_csr(&csr_matrix)?;

        Ok(Self { inner: sym_csr })
    }

    /// Get the underlying matrix
    ///
    /// # Returns
    ///
    /// Reference to the inner SymCSR matrix
    pub fn inner(&self) -> &SymCsrMatrix<T> {
        &self.inner
    }

    /// Get access to the underlying data array
    ///
    /// # Returns
    ///
    /// Reference to the data array
    pub fn data(&self) -> &[T] {
        &self.inner.data
    }

    /// Get access to the underlying indices array
    ///
    /// # Returns
    ///
    /// Reference to the indices array
    pub fn indices(&self) -> &[usize] {
        &self.inner.indices
    }

    /// Get access to the underlying indptr array
    ///
    /// # Returns
    ///
    /// Reference to the indptr array
    pub fn indptr(&self) -> &[usize] {
        &self.inner.indptr
    }

    /// Convert to a standard CSR array
    ///
    /// # Returns
    ///
    /// CSR array containing the full symmetric matrix
    pub fn to_csr_array(&self) -> SparseResult<CsrArray<T>> {
        let csr = self.inner.to_csr()?;

        // Convert the CsrMatrix to CsrArray using from_triplets
        let (rows, cols, data) = csr.get_triplets();
        let shape = csr.shape();

        // Safety check - rows, cols, and data should all be the same length
        if rows.len() != cols.len() || rows.len() != data.len() {
            return Err(SparseError::DimensionMismatch {
                expected: rows.len(),
                found: cols.len().min(data.len()),
            });
        }

        CsrArray::from_triplets(&rows, &cols, &data, shape, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparray::SparseArray;

    #[test]
    fn test_sym_csr_creation() {
        // Create a simple symmetric matrix stored in lower triangular format
        // [2 1 0]
        // [1 2 3]
        // [0 3 1]

        // Note: Actually represents the lower triangular part only:
        // [2 0 0]
        // [1 2 0]
        // [0 3 1]

        let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let indices = vec![0, 0, 1, 1, 2];
        let indptr = vec![0, 1, 3, 5];

        let sym = SymCsrMatrix::new(data, indptr, indices, (3, 3)).unwrap();

        assert_eq!(sym.shape(), (3, 3));
        assert_eq!(sym.nnz_stored(), 5);

        // Total non-zeros should count off-diagonal elements twice
        assert_eq!(sym.nnz(), 7);

        // Test accessing elements
        assert_eq!(sym.get(0, 0), 2.0);
        assert_eq!(sym.get(0, 1), 1.0);
        assert_eq!(sym.get(1, 0), 1.0); // From symmetry
        assert_eq!(sym.get(1, 1), 2.0);
        assert_eq!(sym.get(1, 2), 3.0);
        assert_eq!(sym.get(2, 1), 3.0); // From symmetry
        assert_eq!(sym.get(2, 2), 1.0);
        assert_eq!(sym.get(0, 2), 0.0); // Zero element - not stored
        assert_eq!(sym.get(2, 0), 0.0); // Zero element - not stored
    }

    #[test]
    fn test_sym_csr_from_standard() {
        // Create a standard CSR matrix that's symmetric
        // [2 1 0]
        // [1 2 3]
        // [0 3 1]

        // Create it from triplets to ensure it's properly constructed
        let row_indices = vec![0, 0, 1, 1, 1, 2, 2];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let data = vec![2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0];

        let csr = CsrMatrix::new(data, row_indices, col_indices, (3, 3)).unwrap();
        let sym = SymCsrMatrix::from_csr(&csr).unwrap();

        assert_eq!(sym.shape(), (3, 3));

        // Convert back to standard CSR to check
        let csr2 = sym.to_csr().unwrap();
        let dense = csr2.to_dense();

        // Check the full matrix
        assert_eq!(dense[0][0], 2.0);
        assert_eq!(dense[0][1], 1.0);
        assert_eq!(dense[0][2], 0.0);
        assert_eq!(dense[1][0], 1.0);
        assert_eq!(dense[1][1], 2.0);
        assert_eq!(dense[1][2], 3.0);
        assert_eq!(dense[2][0], 0.0);
        assert_eq!(dense[2][1], 3.0);
        assert_eq!(dense[2][2], 1.0);
    }

    #[test]
    fn test_sym_csr_array() {
        // Create a symmetric SymCSR matrix, storing only the lower triangular part
        let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let indices = vec![0, 0, 1, 1, 2];
        let indptr = vec![0, 1, 3, 5];

        let sym_matrix = SymCsrMatrix::new(data, indptr, indices, (3, 3)).unwrap();
        let sym_array = SymCsrArray::new(sym_matrix);

        assert_eq!(sym_array.inner().shape(), (3, 3));

        // Convert to standard CSR array
        let csr_array = sym_array.to_csr_array().unwrap();

        // Verify shape and values
        assert_eq!(csr_array.shape(), (3, 3));
        assert_eq!(csr_array.get(0, 0), 2.0);
        assert_eq!(csr_array.get(0, 1), 1.0);
        assert_eq!(csr_array.get(1, 0), 1.0);
        assert_eq!(csr_array.get(1, 1), 2.0);
        assert_eq!(csr_array.get(1, 2), 3.0);
        assert_eq!(csr_array.get(2, 1), 3.0);
        assert_eq!(csr_array.get(2, 2), 1.0);
        assert_eq!(csr_array.get(0, 2), 0.0);
        assert_eq!(csr_array.get(2, 0), 0.0);
    }
}
