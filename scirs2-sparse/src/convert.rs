//! Conversion utilities for sparse matrices
//!
//! This module provides functions for converting between different sparse matrix
//! formats and between sparse and dense representations.

use crate::coo::CooMatrix;
use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use crate::error::SparseResult;
use scirs2_core::ndarray::Array2;

/// Convert a dense matrix to CSR format
///
/// # Arguments
///
/// * `dense` - Dense matrix as 2D array
///
/// # Returns
///
/// * Sparse matrix in CSR format
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_sparse::convert::dense_to_csr;
///
/// let dense = Array2::from_shape_vec((3, 3), vec![
///     1.0, 0.0, 2.0,
///     0.0, 0.0, 3.0,
///     4.0, 5.0, 0.0,
/// ]).unwrap();
///
/// let sparse = dense_to_csr(&dense).unwrap();
/// ```
#[allow(dead_code)]
pub fn dense_to_csr(dense: &Array2<f64>) -> SparseResult<CsrMatrix<f64>> {
    let shape = dense.dim();
    let (rows, cols) = (shape.0, shape.1);

    let mut data = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for i in 0..rows {
        for j in 0..cols {
            let val = dense[[i, j]];
            if val != 0.0 {
                data.push(val);
                row_indices.push(i);
                col_indices.push(j);
            }
        }
    }

    CsrMatrix::new(data, row_indices, col_indices, (rows, cols))
}

/// Convert a dense matrix to CSC format
///
/// # Arguments
///
/// * `dense` - Dense matrix as 2D array
///
/// # Returns
///
/// * Sparse matrix in CSC format
#[allow(dead_code)]
pub fn dense_to_csc(dense: &Array2<f64>) -> SparseResult<CscMatrix<f64>> {
    let shape = dense.dim();
    let (rows, cols) = (shape.0, shape.1);

    let mut data = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for i in 0..rows {
        for j in 0..cols {
            let val = dense[[i, j]];
            if val != 0.0 {
                data.push(val);
                row_indices.push(i);
                col_indices.push(j);
            }
        }
    }

    CscMatrix::new(data, row_indices, col_indices, (rows, cols))
}

/// Convert a dense matrix to COO format
///
/// # Arguments
///
/// * `dense` - Dense matrix as 2D array
///
/// # Returns
///
/// * Sparse matrix in COO format
#[allow(dead_code)]
pub fn dense_to_coo(dense: &Array2<f64>) -> SparseResult<CooMatrix<f64>> {
    let shape = dense.dim();
    let (rows, cols) = (shape.0, shape.1);

    let mut data = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for i in 0..rows {
        for j in 0..cols {
            let val = dense[[i, j]];
            if val != 0.0 {
                data.push(val);
                row_indices.push(i);
                col_indices.push(j);
            }
        }
    }

    CooMatrix::new(data, row_indices, col_indices, (rows, cols))
}

/// Convert a CSR matrix to dense format
///
/// # Arguments
///
/// * `sparse` - Sparse matrix in CSR format
///
/// # Returns
///
/// * Dense matrix as 2D array
#[allow(dead_code)]
pub fn csr_to_dense(sparse: &CsrMatrix<f64>) -> Array2<f64> {
    let (rows, cols) = sparse.shape();
    let mut dense = Array2::zeros((rows, cols));

    let dense_vec = sparse.to_dense();
    for i in 0..rows {
        for j in 0..cols {
            dense[[i, j]] = dense_vec[i][j];
        }
    }

    dense
}

/// Convert a CSC matrix to dense format
///
/// # Arguments
///
/// * `sparse` - Sparse matrix in CSC format
///
/// # Returns
///
/// * Dense matrix as 2D array
#[allow(dead_code)]
pub fn csc_to_dense(sparse: &CscMatrix<f64>) -> Array2<f64> {
    let (rows, cols) = sparse.shape();
    let mut dense = Array2::zeros((rows, cols));

    let dense_vec = sparse.to_dense();
    for i in 0..rows {
        for j in 0..cols {
            dense[[i, j]] = dense_vec[i][j];
        }
    }

    dense
}

/// Convert a COO matrix to dense format
///
/// # Arguments
///
/// * `sparse` - Sparse matrix in COO format
///
/// # Returns
///
/// * Dense matrix as 2D array
#[allow(dead_code)]
pub fn coo_to_dense(sparse: &CooMatrix<f64>) -> Array2<f64> {
    let (rows, cols) = sparse.shape();
    let mut dense = Array2::zeros((rows, cols));

    let dense_vec = sparse.to_dense();
    for i in 0..rows {
        for j in 0..cols {
            dense[[i, j]] = dense_vec[i][j];
        }
    }

    dense
}

/// Convert a CSR matrix to COO format
///
/// # Arguments
///
/// * `csr` - Sparse matrix in CSR format
///
/// # Returns
///
/// * Sparse matrix in COO format
#[allow(dead_code)]
pub fn csr_to_coo(csr: &CsrMatrix<f64>) -> CooMatrix<f64> {
    let (rows, cols) = csr.shape();
    let dense = csr.to_dense();

    let mut data = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for (i, row) in dense.iter().enumerate().take(rows) {
        for (j, &val) in row.iter().enumerate().take(cols) {
            if val != 0.0 {
                data.push(val);
                row_indices.push(i);
                col_indices.push(j);
            }
        }
    }

    CooMatrix::new(data, row_indices, col_indices, (rows, cols)).unwrap()
}

/// Convert a CSC matrix to COO format
///
/// # Arguments
///
/// * `csc` - Sparse matrix in CSC format
///
/// # Returns
///
/// * Sparse matrix in COO format
#[allow(dead_code)]
pub fn csc_to_coo(csc: &CscMatrix<f64>) -> CooMatrix<f64> {
    let (rows, cols) = csc.shape();
    let dense = csc.to_dense();

    let mut data = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for (i, row) in dense.iter().enumerate().take(rows) {
        for (j, &val) in row.iter().enumerate().take(cols) {
            if val != 0.0 {
                data.push(val);
                row_indices.push(i);
                col_indices.push(j);
            }
        }
    }

    CooMatrix::new(data, row_indices, col_indices, (rows, cols)).unwrap()
}

/// Convert a COO matrix to CSR format
///
/// # Arguments
///
/// * `coo` - Sparse matrix in COO format
///
/// # Returns
///
/// * Sparse matrix in CSR format
#[allow(dead_code)]
pub fn coo_to_csr(coo: &CooMatrix<f64>) -> CsrMatrix<f64> {
    let (rows, cols) = coo.shape();
    let dense = coo.to_dense();

    let mut data = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for (i, row) in dense.iter().enumerate().take(rows) {
        for (j, &val) in row.iter().enumerate().take(cols) {
            if val != 0.0 {
                data.push(val);
                row_indices.push(i);
                col_indices.push(j);
            }
        }
    }

    CsrMatrix::new(data, row_indices, col_indices, (rows, cols)).unwrap()
}

/// Convert a COO matrix to CSC format
///
/// # Arguments
///
/// * `coo` - Sparse matrix in COO format
///
/// # Returns
///
/// * Sparse matrix in CSC format
#[allow(dead_code)]
pub fn coo_to_csc(coo: &CooMatrix<f64>) -> CscMatrix<f64> {
    let (rows, cols) = coo.shape();
    let dense = coo.to_dense();

    let mut data = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for (i, row) in dense.iter().enumerate().take(rows) {
        for (j, &val) in row.iter().enumerate().take(cols) {
            if val != 0.0 {
                data.push(val);
                row_indices.push(i);
                col_indices.push(j);
            }
        }
    }

    CscMatrix::new(data, row_indices, col_indices, (rows, cols)).unwrap()
}

/// Convert a CSR matrix to CSC format
///
/// # Arguments
///
/// * `csr` - Sparse matrix in CSR format
///
/// # Returns
///
/// * Sparse matrix in CSC format
#[allow(dead_code)]
pub fn csr_to_csc<F>(csr: &CsrMatrix<F>) -> SparseResult<CscMatrix<F>>
where
    F: Clone + Copy + std::fmt::Debug + PartialEq + scirs2_core::numeric::Zero,
{
    // Start with CSR in triplet format
    let (rows, cols) = csr.shape();
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    // Extract all non-zero entries from CSR into COO format triplets
    for i in 0..rows {
        for j in csr.indptr[i]..csr.indptr[i + 1] {
            if j < csr.indices.len() {
                let col = csr.indices[j];
                let val = csr.data[j];

                row_indices.push(i);
                col_indices.push(col);
                values.push(val);
            }
        }
    }

    // Create a COO matrix from the triplets
    let coo = CooMatrix::new(values, row_indices, col_indices, (rows, cols))?;

    // Convert COO to CSC (which basically just sorts by column, then row)
    Ok(coo.to_csc())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_dense_to_csr_to_dense() {
        // Create a dense matrix
        let dense =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0])
                .unwrap();

        // Convert to CSR
        let csr = dense_to_csr(&dense).unwrap();

        // Convert back to dense
        let dense2 = csr_to_dense(&csr);

        // Check equality
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(dense[[i, j]], dense2[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_dense_to_csc_to_dense() {
        // Create a dense matrix
        let dense =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0])
                .unwrap();

        // Convert to CSC
        let csc = dense_to_csc(&dense).unwrap();

        // Convert back to dense
        let dense2 = csc_to_dense(&csc);

        // Check equality
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(dense[[i, j]], dense2[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_dense_to_coo_to_dense() {
        // Create a dense matrix
        let dense =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0])
                .unwrap();

        // Convert to COO
        let coo = dense_to_coo(&dense).unwrap();

        // Convert back to dense
        let dense2 = coo_to_dense(&coo);

        // Check equality
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(dense[[i, j]], dense2[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_format_conversions() {
        // Create a COO matrix
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let coo = CooMatrix::new(data, rows, cols, shape).unwrap();

        // Convert to CSR
        let csr = coo_to_csr(&coo);

        // Convert CSR to COO
        let coo2 = csr_to_coo(&csr);

        // Check that the conversions preserved the data
        let dense1 = coo.to_dense();
        let dense2 = coo2.to_dense();

        assert_eq!(dense1, dense2);
    }
}
