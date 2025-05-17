//! Symmetric Coordinate (SymCOO) module
//!
//! This module provides a specialized implementation of the COO format
//! optimized for symmetric matrices, storing only the lower or upper
//! triangular part of the matrix.

use crate::coo::CooMatrix;
use crate::coo_array::CooArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use num_traits::Float;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

/// Symmetric Coordinate (SymCOO) matrix
///
/// This format stores only the lower triangular part of a symmetric matrix
/// to save memory. It's particularly useful for construction of symmetric
/// matrices and for conversion to other symmetric formats.
///
/// # Note
///
/// All operations maintain symmetry implicitly.
#[derive(Debug, Clone)]
pub struct SymCooMatrix<T>
where
    T: Float + Debug + Copy,
{
    /// Non-zero values in the lower triangular part
    pub data: Vec<T>,

    /// Row indices for each non-zero element
    pub rows: Vec<usize>,

    /// Column indices for each non-zero element
    pub cols: Vec<usize>,

    /// Matrix shape (rows, cols), always square
    pub shape: (usize, usize),
}

impl<T> SymCooMatrix<T>
where
    T: Float + Debug + Copy,
{
    /// Create a new symmetric COO matrix from raw data
    ///
    /// # Arguments
    ///
    /// * `data` - Non-zero values in the lower triangular part
    /// * `rows` - Row indices
    /// * `cols` - Column indices
    /// * `shape` - Matrix shape (n, n)
    ///
    /// # Returns
    ///
    /// A symmetric COO matrix
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The shape is not square
    /// - The arrays have inconsistent lengths
    /// - Any index is out of bounds
    /// - Upper triangular elements are included
    pub fn new(
        data: Vec<T>,
        rows: Vec<usize>,
        cols: Vec<usize>,
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        let (nrows, ncols) = shape;

        // Ensure matrix is square
        if nrows != ncols {
            return Err(SparseError::ValueError(
                "Symmetric matrix must be square".to_string(),
            ));
        }

        // Check array lengths
        let nnz = data.len();
        if rows.len() != nnz || cols.len() != nnz {
            return Err(SparseError::ValueError(format!(
                "Data ({}), row ({}) and column ({}) arrays must have same length",
                nnz,
                rows.len(),
                cols.len()
            )));
        }

        // Check bounds and ensure only lower triangular elements
        for i in 0..nnz {
            let row = rows[i];
            let col = cols[i];

            if row >= nrows {
                return Err(SparseError::IndexOutOfBounds {
                    index: (row, 0),
                    shape: (nrows, ncols),
                });
            }

            if col >= ncols {
                return Err(SparseError::IndexOutOfBounds {
                    index: (row, col),
                    shape: (nrows, ncols),
                });
            }

            // For symmetric storage, we only keep the lower triangular part
            if col > row {
                return Err(SparseError::ValueError(
                    "Symmetric COO should only store the lower triangular part".to_string(),
                ));
            }
        }

        Ok(Self {
            data,
            rows,
            cols,
            shape,
        })
    }

    /// Convert a regular COO matrix to symmetric COO format
    ///
    /// This will verify that the matrix is symmetric and extract
    /// the lower triangular part.
    ///
    /// # Arguments
    ///
    /// * `matrix` - COO matrix to convert
    ///
    /// # Returns
    ///
    /// A symmetric COO matrix
    pub fn from_coo(matrix: &CooMatrix<T>) -> SparseResult<Self> {
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
                "Matrix must be symmetric to convert to SymCOO format".to_string(),
            ));
        }

        // Extract the lower triangular part
        let mut data = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        let rows_vec = matrix.row_indices();
        let cols_vec = matrix.col_indices();
        let data_vec = matrix.data();

        for i in 0..data_vec.len() {
            let row = rows_vec[i];
            let col = cols_vec[i];

            // Only include elements in lower triangular part (including diagonal)
            if col <= row {
                data.push(data_vec[i]);
                row_indices.push(row);
                col_indices.push(col);
            }
        }

        Ok(Self {
            data,
            rows: row_indices,
            cols: col_indices,
            shape: (rows, cols),
        })
    }

    /// Check if a COO matrix is symmetric
    ///
    /// # Arguments
    ///
    /// * `matrix` - COO matrix to check
    ///
    /// # Returns
    ///
    /// `true` if the matrix is symmetric, `false` otherwise
    pub fn is_symmetric(matrix: &CooMatrix<T>) -> bool {
        let (rows, cols) = matrix.shape();

        // Must be square
        if rows != cols {
            return false;
        }

        // Convert to dense to check symmetry (more efficient for COO format)
        let dense = matrix.to_dense();

        for i in 0..rows {
            for j in 0..i {
                // Only need to check upper triangular elements
                // Compare with sufficient tolerance for floating point comparisons
                let diff = (dense[i][j] - dense[j][i]).abs();
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
        let mut count = 0;

        for i in 0..self.data.len() {
            let row = self.rows[i];
            let col = self.cols[i];

            if row == col {
                // Diagonal element, count once
                count += 1;
            } else {
                // Off-diagonal element, count twice (for both triangular parts)
                count += 2;
            }
        }

        count
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

        // Search for the element in COO format
        for i in 0..self.data.len() {
            if self.rows[i] == actual_row && self.cols[i] == actual_col {
                return self.data[i];
            }
        }

        T::zero()
    }

    /// Convert to standard COO matrix (reconstructing full symmetric matrix)
    ///
    /// # Returns
    ///
    /// A standard COO matrix with both upper and lower triangular parts
    pub fn to_coo(&self) -> SparseResult<CooMatrix<T>> {
        let mut data = Vec::new();
        let mut rows = Vec::new();
        let mut cols = Vec::new();

        // Add the stored lower triangular elements
        data.extend_from_slice(&self.data);
        rows.extend_from_slice(&self.rows);
        cols.extend_from_slice(&self.cols);

        // Add the upper triangular elements by symmetry
        for i in 0..self.data.len() {
            let row = self.rows[i];
            let col = self.cols[i];

            // Skip diagonal elements (already included)
            if row != col {
                // Add the symmetric element
                data.push(self.data[i]);
                rows.push(col);
                cols.push(row);
            }
        }

        CooMatrix::new(data, rows, cols, self.shape)
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
        for i in 0..self.data.len() {
            let row = self.rows[i];
            let col = self.cols[i];
            dense[row][col] = self.data[i];

            // Fill the upper triangular part (from symmetry)
            if row != col {
                dense[col][row] = self.data[i];
            }
        }

        dense
    }
}

/// Array-based SymCOO implementation compatible with SparseArray trait
#[derive(Debug, Clone)]
pub struct SymCooArray<T>
where
    T: Float + Debug + Copy,
{
    /// Inner matrix
    inner: SymCooMatrix<T>,
}

impl<T> SymCooArray<T>
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
    /// Create a new SymCOO array from a SymCOO matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - Symmetric COO matrix
    ///
    /// # Returns
    ///
    /// SymCOO array
    pub fn new(matrix: SymCooMatrix<T>) -> Self {
        Self { inner: matrix }
    }

    /// Create a SymCOO array from triplets (row, col, value)
    ///
    /// # Arguments
    ///
    /// * `rows` - Row indices
    /// * `cols` - Column indices
    /// * `data` - Non-zero values
    /// * `shape` - Matrix shape (n, n)
    /// * `enforce_symmetric` - If true, enforce that matrix is symmetric by averaging a_ij and a_ji
    ///
    /// # Returns
    ///
    /// A symmetric COO array
    pub fn from_triplets(
        rows: &[usize],
        cols: &[usize],
        data: &[T],
        shape: (usize, usize),
        enforce_symmetric: bool,
    ) -> SparseResult<Self> {
        if shape.0 != shape.1 {
            return Err(SparseError::ValueError(
                "Symmetric matrix must be square".to_string(),
            ));
        }

        if !enforce_symmetric {
            // Create a temporary dense matrix to check symmetry
            let n = shape.0;
            let mut dense = vec![vec![T::zero(); n]; n];
            let nnz = data.len().min(rows.len().min(cols.len()));

            // Fill the matrix with the provided elements
            for i in 0..nnz {
                let row = rows[i];
                let col = cols[i];

                if row >= n || col >= n {
                    return Err(SparseError::IndexOutOfBounds {
                        index: (row, col),
                        shape,
                    });
                }

                dense[row][col] = data[i];
            }

            // Check if the matrix is symmetric
            for i in 0..n {
                for j in 0..i {
                    if (dense[i][j] - dense[j][i]).abs() > T::epsilon() {
                        return Err(SparseError::ValueError(
                            "Input is not symmetric. Use enforce_symmetric=true to force symmetry"
                                .to_string(),
                        ));
                    }
                }
            }

            // Extract lower triangular part
            let mut sym_data = Vec::new();
            let mut sym_rows = Vec::new();
            let mut sym_cols = Vec::new();

            for (i, row) in dense.iter().enumerate().take(n) {
                for j in 0..=i {
                    let val = row[j];
                    if val != T::zero() {
                        sym_data.push(val);
                        sym_rows.push(i);
                        sym_cols.push(j);
                    }
                }
            }

            // Create the symmetric matrix
            let sym_coo = SymCooMatrix::new(sym_data, sym_rows, sym_cols, shape)?;
            return Ok(Self { inner: sym_coo });
        }

        // Create a symmetric matrix by averaging corresponding elements
        let n = shape.0;

        // First, build a dense matrix with all input elements
        let mut dense = vec![vec![T::zero(); n]; n];
        let nnz = data.len();

        // Add all elements to the matrix
        for i in 0..nnz {
            if i >= rows.len() || i >= cols.len() {
                return Err(SparseError::ValueError(
                    "Inconsistent input arrays".to_string(),
                ));
            }

            let row = rows[i];
            let col = cols[i];

            if row >= n || col >= n {
                return Err(SparseError::IndexOutOfBounds {
                    index: (row, col),
                    shape: (n, n),
                });
            }

            dense[row][col] = data[i];
        }

        // Make symmetric by averaging a_ij and a_ji
        for i in 0..n {
            for j in 0..i {
                let avg = (dense[i][j] + dense[j][i]) / (T::one() + T::one());
                dense[i][j] = avg;
                dense[j][i] = avg;
            }
        }

        // Extract the lower triangular part to create SymCOO
        let mut sym_data = Vec::new();
        let mut sym_rows = Vec::new();
        let mut sym_cols = Vec::new();

        for (i, row) in dense.iter().enumerate().take(n) {
            for j in 0..=i {
                let val = row[j];
                if val != T::zero() {
                    sym_data.push(val);
                    sym_rows.push(i);
                    sym_cols.push(j);
                }
            }
        }

        let sym_coo = SymCooMatrix::new(sym_data, sym_rows, sym_cols, shape)?;
        Ok(Self { inner: sym_coo })
    }

    /// Create a SymCOO array from a regular COO array
    ///
    /// # Arguments
    ///
    /// * `array` - COO array to convert
    ///
    /// # Returns
    ///
    /// A symmetric COO array
    pub fn from_coo_array(array: &CooArray<T>) -> SparseResult<Self> {
        let shape = array.shape();
        let (rows, cols) = shape;

        // Ensure matrix is square
        if rows != cols {
            return Err(SparseError::ValueError(
                "Symmetric matrix must be square".to_string(),
            ));
        }

        // Create a temporary COO matrix to check symmetry
        let coo_matrix = CooMatrix::new(
            array.get_data().to_vec(),
            array.get_rows().to_vec(),
            array.get_cols().to_vec(),
            shape,
        )?;

        // Convert to symmetric COO
        let sym_coo = SymCooMatrix::from_coo(&coo_matrix)?;

        Ok(Self { inner: sym_coo })
    }

    /// Get the underlying matrix
    ///
    /// # Returns
    ///
    /// Reference to the inner SymCOO matrix
    pub fn inner(&self) -> &SymCooMatrix<T> {
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

    /// Get access to the underlying rows array
    ///
    /// # Returns
    ///
    /// Reference to the rows array
    pub fn rows(&self) -> &[usize] {
        &self.inner.rows
    }

    /// Get access to the underlying cols array
    ///
    /// # Returns
    ///
    /// Reference to the cols array
    pub fn cols(&self) -> &[usize] {
        &self.inner.cols
    }

    /// Get the shape of the array
    ///
    /// # Returns
    ///
    /// A tuple (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape
    }

    /// Convert to a standard COO array
    ///
    /// # Returns
    ///
    /// COO array containing the full symmetric matrix
    pub fn to_coo_array(&self) -> SparseResult<CooArray<T>> {
        let coo = self.inner.to_coo()?;

        // Get triplets from CooMatrix
        let rows = coo.row_indices();
        let cols = coo.col_indices();
        let data = coo.data();

        // Create CooArray from triplets
        CooArray::from_triplets(rows, cols, data, coo.shape(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparray::SparseArray;

    #[test]
    fn test_sym_coo_creation() {
        // Create a simple symmetric matrix stored in lower triangular format
        // [2 1 0]
        // [1 2 3]
        // [0 3 1]

        let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let rows = vec![0, 1, 1, 2, 2];
        let cols = vec![0, 0, 1, 1, 2];

        let sym = SymCooMatrix::new(data, rows, cols, (3, 3)).unwrap();

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
        assert_eq!(sym.get(0, 2), 0.0);
        assert_eq!(sym.get(2, 0), 0.0);
    }

    #[test]
    fn test_sym_coo_from_standard() {
        // Create a standard COO matrix that's symmetric
        // [2 1 0]
        // [1 2 3]
        // [0 3 1]

        let data = vec![2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0];
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];

        let coo = CooMatrix::new(data, rows, cols, (3, 3)).unwrap();
        let sym = SymCooMatrix::from_coo(&coo).unwrap();

        assert_eq!(sym.shape(), (3, 3));

        // Convert back to standard COO
        let coo2 = sym.to_coo().unwrap();
        let dense = coo2.to_dense();

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
    fn test_sym_coo_array() {
        // Create a symmetric SymCOO matrix
        let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let rows = vec![0, 1, 1, 2, 2];
        let cols = vec![0, 0, 1, 1, 2];

        let sym_matrix = SymCooMatrix::new(data, rows, cols, (3, 3)).unwrap();
        let sym_array = SymCooArray::new(sym_matrix);

        assert_eq!(sym_array.inner().shape(), (3, 3));

        // Convert to standard COO array
        let coo_array = sym_array.to_coo_array().unwrap();

        // Verify shape and values
        assert_eq!(coo_array.shape(), (3, 3));
        assert_eq!(coo_array.get(0, 0), 2.0);
        assert_eq!(coo_array.get(0, 1), 1.0);
        assert_eq!(coo_array.get(1, 0), 1.0);
        assert_eq!(coo_array.get(1, 1), 2.0);
        assert_eq!(coo_array.get(1, 2), 3.0);
        assert_eq!(coo_array.get(2, 1), 3.0);
        assert_eq!(coo_array.get(2, 2), 1.0);
        assert_eq!(coo_array.get(0, 2), 0.0);
        assert_eq!(coo_array.get(2, 0), 0.0);
    }

    #[test]
    fn test_sym_coo_array_from_triplets() {
        // Test creating a symmetric matrix from triplets
        // This needs to be a truly symmetric matrix to work without enforce_symmetric
        let rows = vec![0, 1, 1, 2, 1, 0, 2];
        let cols = vec![0, 1, 2, 2, 0, 1, 1];
        let data = vec![2.0, 2.0, 3.0, 1.0, 1.0, 1.0, 3.0];

        let sym_array = SymCooArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        assert_eq!(sym_array.shape(), (3, 3));

        // Test with enforcement of symmetry (add asymmetric values)
        let rows2 = vec![0, 0, 1, 1, 2, 1];
        let cols2 = vec![0, 1, 1, 2, 2, 0];
        let data2 = vec![2.0, 1.0, 2.0, 3.0, 1.0, 2.0]; // 1,0 element is 2.0 instead of 1.0

        let sym_array2 = SymCooArray::from_triplets(&rows2, &cols2, &data2, (3, 3), true).unwrap();

        // Should average the (1,0) and (0,1) elements to 1.5
        assert_eq!(sym_array2.inner().get(1, 0), 1.5);
        assert_eq!(sym_array2.inner().get(0, 1), 1.5);
    }
}
