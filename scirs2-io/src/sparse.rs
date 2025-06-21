//! Comprehensive sparse matrix format support
//!
//! This module provides unified support for common sparse matrix formats used in
//! scientific computing: COO (Coordinate), CSR (Compressed Sparse Row), and
//! CSC (Compressed Sparse Column). It integrates and enhances the existing
//! sparse matrix functionality from other modules.
//!
//! ## Features
//!
//! - **Multiple Formats**: Support for COO, CSR, and CSC formats
//! - **Format Conversion**: Efficient conversion between formats
//! - **Matrix Operations**: Basic sparse matrix operations (addition, multiplication, transpose)
//! - **I/O Support**: Reading and writing in various file formats
//! - **Integration**: Seamless integration with Matrix Market and other formats
//! - **Performance**: Optimized algorithms for large sparse matrices
//! - **Memory Efficiency**: Minimal memory overhead for sparse data
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::sparse::{SparseMatrix, SparseFormat};
//! use ndarray::Array2;
//!
//! // Create a sparse matrix from a dense array
//! let dense = Array2::from_shape_vec((3, 3), vec![
//!     1.0, 0.0, 2.0,
//!     0.0, 3.0, 0.0,
//!     4.0, 0.0, 5.0
//! ]).unwrap();
//!
//! let mut sparse = SparseMatrix::from_dense(&dense, 1e-10)?;
//! println!("Sparse matrix: {} non-zeros", sparse.nnz());
//!
//! // Convert to different formats
//! let csr = sparse.to_csr()?;
//! let csc = sparse.to_csc()?;
//!
//! // Save to file
//! sparse.save_matrix_market("matrix.mtx")?;
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

use crate::error::{IoError, Result};
use crate::matrix_market::{
    MMDataType, MMFormat, MMHeader, MMSparseMatrix, MMSymmetry, SparseEntry,
};
use crate::serialize::{SparseMatrixCOO, SparseMatrixCSC, SparseMatrixCSR};
use ndarray::{Array2, ArrayBase, Data, Dimension};
use std::collections::HashMap;
use std::path::Path;

/// Comprehensive sparse matrix supporting multiple formats
#[derive(Debug, Clone)]
pub struct SparseMatrix<T> {
    /// Matrix dimensions (rows, cols)
    pub shape: (usize, usize),
    /// Number of non-zero elements
    pub nnz: usize,
    /// Primary format for storage
    pub format: SparseFormat,
    /// COO format data (always available)
    pub coo: SparseMatrixCOO<T>,
    /// CSR format data (computed on demand)
    pub csr: Option<SparseMatrixCSR<T>>,
    /// CSC format data (computed on demand)
    pub csc: Option<SparseMatrixCSC<T>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Sparse matrix storage formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Coordinate format (triplets: row, col, value)
    COO,
    /// Compressed Sparse Row format
    CSR,
    /// Compressed Sparse Column format
    CSC,
}

/// Sparse matrix statistics for analysis
#[derive(Debug, Clone)]
pub struct SparseStats {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Sparsity ratio (nnz / (rows * cols))
    pub density: f64,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Average non-zeros per row
    pub avg_nnz_per_row: f64,
    /// Average non-zeros per column
    pub avg_nnz_per_col: f64,
    /// Maximum non-zeros in any row
    pub max_nnz_row: usize,
    /// Maximum non-zeros in any column
    pub max_nnz_col: usize,
}

impl<T> SparseMatrix<T>
where
    T: Clone + Default + PartialEq,
{
    /// Create a new empty sparse matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            shape: (rows, cols),
            nnz: 0,
            format: SparseFormat::COO,
            coo: SparseMatrixCOO::new(rows, cols),
            csr: None,
            csc: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a sparse matrix from COO data
    pub fn from_coo(coo: SparseMatrixCOO<T>) -> Self {
        let nnz = coo.values.len();
        Self {
            shape: (coo.rows, coo.cols),
            nnz,
            format: SparseFormat::COO,
            coo,
            csr: None,
            csc: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a sparse matrix from triplets (row, col, value)
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self> {
        if row_indices.len() != col_indices.len() || col_indices.len() != values.len() {
            return Err(IoError::ValidationError(
                "Triplet arrays must have the same length".to_string(),
            ));
        }

        let nnz = values.len();
        let coo = SparseMatrixCOO {
            rows,
            cols,
            row_indices,
            col_indices,
            values,
            metadata: HashMap::new(),
        };

        Ok(Self {
            shape: (rows, cols),
            nnz,
            format: SparseFormat::COO,
            coo,
            csr: None,
            csc: None,
            metadata: HashMap::new(),
        })
    }

    /// Add a non-zero element to the matrix
    pub fn push(&mut self, row: usize, col: usize, value: T) -> Result<()> {
        if row >= self.shape.0 || col >= self.shape.1 {
            return Err(IoError::ValidationError("Index out of bounds".to_string()));
        }

        self.coo.push(row, col, value);
        self.nnz += 1;

        // Invalidate cached formats
        self.csr = None;
        self.csc = None;

        Ok(())
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Get matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get sparsity density (nnz / total_elements)
    pub fn density(&self) -> f64 {
        let total = self.shape.0 * self.shape.1;
        if total == 0 {
            0.0
        } else {
            self.nnz as f64 / total as f64
        }
    }

    /// Convert to CSR format (cached)
    pub fn to_csr(&mut self) -> Result<&SparseMatrixCSR<T>> {
        if self.csr.is_none() {
            self.csr = Some(self.convert_to_csr()?);
        }
        Ok(self.csr.as_ref().unwrap())
    }

    /// Convert to CSC format (cached)
    pub fn to_csc(&mut self) -> Result<&SparseMatrixCSC<T>> {
        if self.csc.is_none() {
            self.csc = Some(self.convert_to_csc()?);
        }
        Ok(self.csc.as_ref().unwrap())
    }

    /// Convert to COO format (always available)
    pub fn to_coo(&self) -> &SparseMatrixCOO<T> {
        &self.coo
    }

    /// Convert COO to CSR format
    fn convert_to_csr(&self) -> Result<SparseMatrixCSR<T>> {
        let rows = self.shape.0;
        let nnz = self.nnz;

        if nnz == 0 {
            return Ok(SparseMatrixCSR {
                rows,
                cols: self.shape.1,
                values: Vec::new(),
                col_indices: Vec::new(),
                row_ptrs: vec![0; rows + 1],
                metadata: HashMap::new(),
            });
        }

        // Count elements per row
        let mut row_counts = vec![0; rows];
        for &row in &self.coo.row_indices {
            row_counts[row] += 1;
        }

        // Compute row pointers
        let mut row_ptrs = vec![0; rows + 1];
        for i in 0..rows {
            row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
        }

        // Sort by row, then by column
        let mut triplets: Vec<(usize, usize, &T)> = self
            .coo
            .row_indices
            .iter()
            .zip(&self.coo.col_indices)
            .zip(&self.coo.values)
            .map(|((&row, &col), val)| (row, col, val))
            .collect();

        triplets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Extract values and column indices
        let values: Vec<T> = triplets.iter().map(|(_, _, val)| (*val).clone()).collect();
        let col_indices: Vec<usize> = triplets.iter().map(|(_, col, _)| *col).collect();

        Ok(SparseMatrixCSR {
            rows,
            cols: self.shape.1,
            values,
            col_indices,
            row_ptrs,
            metadata: HashMap::new(),
        })
    }

    /// Convert COO to CSC format
    fn convert_to_csc(&self) -> Result<SparseMatrixCSC<T>> {
        let cols = self.shape.1;
        let nnz = self.nnz;

        if nnz == 0 {
            return Ok(SparseMatrixCSC {
                rows: self.shape.0,
                cols,
                values: Vec::new(),
                row_indices: Vec::new(),
                col_ptrs: vec![0; cols + 1],
                metadata: HashMap::new(),
            });
        }

        // Count elements per column
        let mut col_counts = vec![0; cols];
        for &col in &self.coo.col_indices {
            col_counts[col] += 1;
        }

        // Compute column pointers
        let mut col_ptrs = vec![0; cols + 1];
        for i in 0..cols {
            col_ptrs[i + 1] = col_ptrs[i] + col_counts[i];
        }

        // Sort by column, then by row
        let mut triplets: Vec<(usize, usize, &T)> = self
            .coo
            .row_indices
            .iter()
            .zip(&self.coo.col_indices)
            .zip(&self.coo.values)
            .map(|((&row, &col), val)| (row, col, val))
            .collect();

        triplets.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        // Extract values and row indices
        let values: Vec<T> = triplets.iter().map(|(_, _, val)| (*val).clone()).collect();
        let row_indices: Vec<usize> = triplets.iter().map(|(row, _, _)| *row).collect();

        Ok(SparseMatrixCSC {
            rows: self.shape.0,
            cols,
            values,
            row_indices,
            col_ptrs,
            metadata: HashMap::new(),
        })
    }

    /// Get comprehensive statistics about the sparse matrix
    pub fn stats(&mut self) -> Result<SparseStats> {
        // Store shape values before mutable borrows
        let (rows, cols) = self.shape;
        let nnz = self.nnz;

        let csr = self.to_csr()?;

        // Calculate row statistics
        let mut row_nnz = vec![0; rows];
        for (i, nnz) in row_nnz.iter_mut().enumerate().take(rows) {
            *nnz = csr.row_ptrs[i + 1] - csr.row_ptrs[i];
        }

        let max_nnz_row = row_nnz.iter().max().copied().unwrap_or(0);
        let avg_nnz_per_row = if rows > 0 {
            nnz as f64 / rows as f64
        } else {
            0.0
        };

        // Calculate column statistics
        let csc = self.to_csc()?;
        let mut col_nnz = vec![0; cols];
        for (j, nnz) in col_nnz.iter_mut().enumerate().take(cols) {
            *nnz = csc.col_ptrs[j + 1] - csc.col_ptrs[j];
        }

        let max_nnz_col = col_nnz.iter().max().copied().unwrap_or(0);
        let avg_nnz_per_col = if cols > 0 {
            nnz as f64 / cols as f64
        } else {
            0.0
        };

        // Memory usage estimation
        let coo_size = self.coo.values.len() * std::mem::size_of::<T>()
            + self.coo.row_indices.len() * std::mem::size_of::<usize>()
            + self.coo.col_indices.len() * std::mem::size_of::<usize>();

        let csr_size = if let Some(ref csr) = self.csr {
            csr.values.len() * std::mem::size_of::<T>()
                + csr.col_indices.len() * std::mem::size_of::<usize>()
                + csr.row_ptrs.len() * std::mem::size_of::<usize>()
        } else {
            0
        };

        let csc_size = if let Some(ref csc) = self.csc {
            csc.values.len() * std::mem::size_of::<T>()
                + csc.row_indices.len() * std::mem::size_of::<usize>()
                + csc.col_ptrs.len() * std::mem::size_of::<usize>()
        } else {
            0
        };

        Ok(SparseStats {
            rows,
            cols,
            nnz,
            density: nnz as f64 / ((rows * cols) as f64),
            memory_bytes: coo_size + csr_size + csc_size,
            avg_nnz_per_row,
            avg_nnz_per_col,
            max_nnz_row,
            max_nnz_col,
        })
    }
}

impl<T> SparseMatrix<T>
where
    T: Clone + Default + PartialEq + PartialOrd,
{
    /// Create a sparse matrix from a dense 2D array with threshold
    pub fn from_dense_2d(array: &Array2<T>, threshold: T) -> Result<Self> {
        let (rows, cols) = array.dim();
        let mut sparse = Self::new(rows, cols);

        // Extract non-zero elements
        for i in 0..rows {
            for j in 0..cols {
                let val = array[[i, j]].clone();
                if val != threshold {
                    sparse.push(i, j, val)?;
                }
            }
        }

        Ok(sparse)
    }

    /// Create a sparse matrix from a dense array with threshold (generic version)
    pub fn from_dense<S, D>(array: &ArrayBase<S, D>, threshold: T) -> Result<Self>
    where
        S: Data<Elem = T>,
        D: Dimension,
    {
        if array.ndim() != 2 {
            return Err(IoError::ValidationError(
                "Only 2D arrays are supported".to_string(),
            ));
        }

        let shape = array.shape();
        let (rows, cols) = (shape[0], shape[1]);
        let mut sparse = Self::new(rows, cols);

        // Extract non-zero elements using flatten with row-major indexing
        for (linear_idx, val) in array.iter().enumerate() {
            if *val != threshold {
                let row = linear_idx / cols;
                let col = linear_idx % cols;
                sparse.push(row, col, val.clone())?;
            }
        }

        Ok(sparse)
    }

    /// Convert sparse matrix to dense array
    pub fn to_dense(&self) -> Array2<T> {
        let mut dense = Array2::default(self.shape);

        for ((row, col), value) in self
            .coo
            .row_indices
            .iter()
            .zip(&self.coo.col_indices)
            .zip(&self.coo.values)
        {
            dense[(*row, *col)] = value.clone();
        }

        dense
    }
}

impl SparseMatrix<f64> {
    /// Load a sparse matrix from a Matrix Market file
    pub fn load_matrix_market<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mm_matrix = crate::matrix_market::read_sparse_matrix(path)?;
        Ok(Self::from_matrix_market(&mm_matrix))
    }

    /// Save the sparse matrix to a Matrix Market file
    pub fn save_matrix_market<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mm_matrix = self.to_matrix_market();
        crate::matrix_market::write_sparse_matrix(path, &mm_matrix)
    }

    /// Create from Matrix Market format
    fn from_matrix_market(mm_matrix: &MMSparseMatrix<f64>) -> Self {
        let mut sparse = Self::new(mm_matrix.rows, mm_matrix.cols);

        for entry in &mm_matrix.entries {
            sparse.push(entry.row, entry.col, entry.value).unwrap();
        }

        // Copy metadata
        sparse
            .metadata
            .insert("format".to_string(), mm_matrix.header.format.to_string());
        sparse.metadata.insert(
            "data_type".to_string(),
            mm_matrix.header.data_type.to_string(),
        );
        sparse.metadata.insert(
            "symmetry".to_string(),
            mm_matrix.header.symmetry.to_string(),
        );

        sparse
    }

    /// Convert to Matrix Market format
    fn to_matrix_market(&self) -> MMSparseMatrix<f64> {
        let header = MMHeader {
            object: "matrix".to_string(),
            format: MMFormat::Coordinate,
            data_type: MMDataType::Real,
            symmetry: MMSymmetry::General,
            comments: vec!["Generated by SciRS2 sparse matrix module".to_string()],
        };

        let entries: Vec<SparseEntry<f64>> = self
            .coo
            .row_indices
            .iter()
            .zip(&self.coo.col_indices)
            .zip(&self.coo.values)
            .map(|((&row, &col), &value)| SparseEntry { row, col, value })
            .collect();

        MMSparseMatrix {
            header,
            rows: self.shape.0,
            cols: self.shape.1,
            nnz: self.nnz,
            entries,
        }
    }

    /// Transpose the sparse matrix
    pub fn transpose(&self) -> Self {
        let mut transposed = Self::new(self.shape.1, self.shape.0);

        for ((row, col), value) in self
            .coo
            .row_indices
            .iter()
            .zip(&self.coo.col_indices)
            .zip(&self.coo.values)
        {
            transposed.push(*col, *row, *value).unwrap();
        }

        transposed
    }
}

/// Sparse matrix operations
pub mod ops {
    use super::*;
    use std::ops::Add;

    impl Add for &SparseMatrix<f64> {
        type Output = Result<SparseMatrix<f64>>;

        fn add(self, other: &SparseMatrix<f64>) -> Self::Output {
            if self.shape != other.shape {
                return Err(IoError::ValidationError(
                    "Matrix dimensions must match for addition".to_string(),
                ));
            }

            let mut result = SparseMatrix::new(self.shape.0, self.shape.1);
            let mut value_map: HashMap<(usize, usize), f64> = HashMap::new();

            // Add values from first matrix
            for ((row, col), value) in self
                .coo
                .row_indices
                .iter()
                .zip(&self.coo.col_indices)
                .zip(&self.coo.values)
            {
                value_map.insert((*row, *col), *value);
            }

            // Add values from second matrix
            for ((row, col), value) in other
                .coo
                .row_indices
                .iter()
                .zip(&other.coo.col_indices)
                .zip(&other.coo.values)
            {
                *value_map.entry((*row, *col)).or_insert(0.0) += *value;
            }

            // Create result matrix
            for ((row, col), value) in value_map {
                if value.abs() > 1e-15 {
                    // Filter out near-zero values
                    result.push(row, col, value)?;
                }
            }

            Ok(result)
        }
    }

    /// Sparse matrix-vector multiplication (CSR format)
    pub fn spmv(matrix: &mut SparseMatrix<f64>, vector: &[f64]) -> Result<Vec<f64>> {
        let (rows, cols) = matrix.shape;

        if cols != vector.len() {
            return Err(IoError::ValidationError(
                "Matrix columns must match vector length".to_string(),
            ));
        }

        let csr = matrix.to_csr()?;
        let mut result = vec![0.0; rows];

        for (i, result_val) in result.iter_mut().enumerate().take(rows) {
            let start = csr.row_ptrs[i];
            let end = csr.row_ptrs[i + 1];

            for idx in start..end {
                let col = csr.col_indices[idx];
                let val = csr.values[idx];
                *result_val += val * vector[col];
            }
        }

        Ok(result)
    }

    /// Sparse matrix-matrix multiplication (simplified version)
    pub fn spmm(a: &mut SparseMatrix<f64>, b: &mut SparseMatrix<f64>) -> Result<SparseMatrix<f64>> {
        let (a_rows, a_cols) = a.shape;
        let (b_rows, b_cols) = b.shape;

        if a_cols != b_rows {
            return Err(IoError::ValidationError(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        let a_csr = a.to_csr()?;
        let b_csc = b.to_csc()?;

        let mut result = SparseMatrix::new(a_rows, b_cols);

        for i in 0..a_rows {
            let a_start = a_csr.row_ptrs[i];
            let a_end = a_csr.row_ptrs[i + 1];

            for j in 0..b_cols {
                let b_start = b_csc.col_ptrs[j];
                let b_end = b_csc.col_ptrs[j + 1];

                let mut dot_product = 0.0;
                let mut a_idx = a_start;
                let mut b_idx = b_start;

                while a_idx < a_end && b_idx < b_end {
                    let a_col = a_csr.col_indices[a_idx];
                    let b_row = b_csc.row_indices[b_idx];

                    match a_col.cmp(&b_row) {
                        std::cmp::Ordering::Equal => {
                            dot_product += a_csr.values[a_idx] * b_csc.values[b_idx];
                            a_idx += 1;
                            b_idx += 1;
                        }
                        std::cmp::Ordering::Less => {
                            a_idx += 1;
                        }
                        std::cmp::Ordering::Greater => {
                            b_idx += 1;
                        }
                    }
                }

                if dot_product.abs() > 1e-15 {
                    result.push(i, j, dot_product)?;
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::tempdir;

    #[test]
    fn test_sparse_matrix_creation() {
        let mut sparse = SparseMatrix::new(3, 3);
        sparse.push(0, 0, 1.0).unwrap();
        sparse.push(1, 1, 2.0).unwrap();
        sparse.push(2, 2, 3.0).unwrap();

        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.shape(), (3, 3));
        assert_eq!(sparse.density(), 3.0 / 9.0);
    }

    #[test]
    fn test_format_conversion() {
        let mut sparse = SparseMatrix::new(3, 3);
        sparse.push(0, 0, 1.0).unwrap();
        sparse.push(0, 2, 2.0).unwrap();
        sparse.push(2, 1, 3.0).unwrap();

        // Test CSR conversion
        let csr = sparse.to_csr().unwrap();
        assert_eq!(csr.row_ptrs, vec![0, 2, 2, 3]);
        assert_eq!(csr.col_indices, vec![0, 2, 1]);
        assert_eq!(csr.values, vec![1.0, 2.0, 3.0]);

        // Test CSC conversion
        let csc = sparse.to_csc().unwrap();
        assert_eq!(csc.col_ptrs, vec![0, 1, 2, 3]);
        assert_eq!(csc.row_indices, vec![0, 2, 0]);
        assert_eq!(csc.values, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn test_dense_sparse_conversion() {
        let dense = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let sparse = SparseMatrix::from_dense(&dense, 0.0).unwrap();

        assert_eq!(sparse.nnz(), 5);

        let reconstructed = sparse.to_dense();
        assert_eq!(dense, reconstructed);
    }

    #[test]
    fn test_matrix_addition() {
        let mut a = SparseMatrix::new(2, 2);
        a.push(0, 0, 1.0).unwrap();
        a.push(1, 1, 2.0).unwrap();

        let mut b = SparseMatrix::new(2, 2);
        b.push(0, 0, 3.0).unwrap();
        b.push(0, 1, 4.0).unwrap();

        let result = (&a + &b).unwrap();
        assert_eq!(result.nnz(), 3);

        let dense_result = result.to_dense();
        let expected = array![[4.0, 4.0], [0.0, 2.0]];
        assert_eq!(dense_result, expected);
    }

    #[test]
    fn test_sparse_matrix_vector_multiplication() {
        let mut sparse = SparseMatrix::new(3, 3);
        sparse.push(0, 0, 1.0).unwrap();
        sparse.push(0, 1, 2.0).unwrap();
        sparse.push(1, 1, 3.0).unwrap();
        sparse.push(2, 2, 4.0).unwrap();

        let vector = vec![1.0, 2.0, 3.0];
        let result = ops::spmv(&mut sparse, &vector).unwrap();

        assert_eq!(result, vec![5.0, 6.0, 12.0]);
    }

    #[test]
    fn test_matrix_market_integration() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.mtx");

        // Create a sparse matrix
        let mut sparse = SparseMatrix::new(3, 3);
        sparse.push(0, 0, 1.5).unwrap();
        sparse.push(1, 1, 2.5).unwrap();
        sparse.push(2, 0, 3.5).unwrap();

        // Save to Matrix Market format
        sparse.save_matrix_market(&file_path).unwrap();

        // Load back
        let loaded = SparseMatrix::load_matrix_market(&file_path).unwrap();

        assert_eq!(loaded.nnz(), 3);
        assert_eq!(loaded.shape(), (3, 3));
        assert_eq!(loaded.to_dense(), sparse.to_dense());
    }

    #[test]
    fn test_sparse_statistics() {
        let mut sparse = SparseMatrix::new(100, 50);

        // Add some elements to create patterns
        for i in 0..10 {
            sparse.push(i, i, 1.0).unwrap();
            if i < 5 {
                sparse.push(i, i + 10, 2.0).unwrap();
            }
        }

        let stats = sparse.stats().unwrap();

        assert_eq!(stats.rows, 100);
        assert_eq!(stats.cols, 50);
        assert_eq!(stats.nnz, 15);
        assert!(stats.density > 0.0 && stats.density < 1.0);
        assert!(stats.memory_bytes > 0);
        assert_eq!(stats.max_nnz_row, 2);
    }

    #[test]
    fn test_transpose() {
        let mut sparse = SparseMatrix::new(2, 3);
        sparse.push(0, 1, 1.0).unwrap();
        sparse.push(1, 2, 2.0).unwrap();

        let transposed = sparse.transpose();

        assert_eq!(transposed.shape(), (3, 2));
        assert_eq!(transposed.nnz(), 2);

        let dense_original = sparse.to_dense();
        let dense_transposed = transposed.to_dense();

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(dense_original[[i, j]], dense_transposed[[j, i]]);
            }
        }
    }
}
