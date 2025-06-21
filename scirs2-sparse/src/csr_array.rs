// CSR Array implementation
//
// This module provides the CSR (Compressed Sparse Row) array format,
// which is efficient for row-wise operations and is one of the most common formats.

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use std::fmt::{self, Debug};
use std::ops::{Add, Div, Mul, Sub};

use crate::error::{SparseError, SparseResult};
use crate::sparray::{SparseArray, SparseSum};

/// CSR Array format
///
/// The CSR (Compressed Sparse Row) format stores a sparse array in three arrays:
/// - data: array of non-zero values
/// - indices: column indices of the non-zero values
/// - indptr: index pointers; for each row, points to the first non-zero element
///
/// # Notes
///
/// - Efficient for row-oriented operations
/// - Fast matrix-vector multiplications
/// - Fast row slicing
/// - Slow column slicing
/// - Slow constructing by setting individual elements
///
#[derive(Clone)]
pub struct CsrArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static,
{
    /// Non-zero values
    data: Array1<T>,
    /// Column indices of non-zero values
    indices: Array1<usize>,
    /// Row pointers (indices into data/indices for the start of each row)
    indptr: Array1<usize>,
    /// Shape of the sparse array
    shape: (usize, usize),
    /// Whether indices are sorted for each row
    has_sorted_indices: bool,
}

impl<T> CsrArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static,
{
    /// Creates a new CSR array from raw components
    ///
    /// # Arguments
    /// * `data` - Array of non-zero values
    /// * `indices` - Column indices of non-zero values
    /// * `indptr` - Index pointers for the start of each row
    /// * `shape` - Shape of the sparse array
    ///
    /// # Returns
    /// A new `CsrArray`
    ///
    /// # Errors
    /// Returns an error if the data is not consistent
    pub fn new(
        data: Array1<T>,
        indices: Array1<usize>,
        indptr: Array1<usize>,
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        // Validation
        if data.len() != indices.len() {
            return Err(SparseError::InconsistentData {
                reason: "data and indices must have the same length".to_string(),
            });
        }

        if indptr.len() != shape.0 + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "indptr length ({}) must be one more than the number of rows ({})",
                    indptr.len(),
                    shape.0
                ),
            });
        }

        if let Some(&max_idx) = indices.iter().max() {
            if max_idx >= shape.1 {
                return Err(SparseError::IndexOutOfBounds {
                    index: (0, max_idx),
                    shape,
                });
            }
        }

        if let Some((&last, &first)) = indptr.iter().next_back().zip(indptr.iter().next()) {
            if first != 0 {
                return Err(SparseError::InconsistentData {
                    reason: "first element of indptr must be 0".to_string(),
                });
            }

            if last != data.len() {
                return Err(SparseError::InconsistentData {
                    reason: format!(
                        "last element of indptr ({}) must equal data length ({})",
                        last,
                        data.len()
                    ),
                });
            }
        }

        let has_sorted_indices = Self::check_sorted_indices(&indices, &indptr);

        Ok(Self {
            data,
            indices,
            indptr,
            shape,
            has_sorted_indices,
        })
    }

    /// Create a CSR array from triplet format (COO-like)
    ///
    /// # Arguments
    /// * `rows` - Row indices
    /// * `cols` - Column indices
    /// * `data` - Values
    /// * `shape` - Shape of the sparse array
    /// * `sorted` - Whether the triplets are sorted by row
    ///
    /// # Returns
    /// A new `CsrArray`
    ///
    /// # Errors
    /// Returns an error if the data is not consistent
    pub fn from_triplets(
        rows: &[usize],
        cols: &[usize],
        data: &[T],
        shape: (usize, usize),
        sorted: bool,
    ) -> SparseResult<Self> {
        if rows.len() != cols.len() || rows.len() != data.len() {
            return Err(SparseError::InconsistentData {
                reason: "rows, cols, and data must have the same length".to_string(),
            });
        }

        if rows.is_empty() {
            // Empty matrix
            let indptr = Array1::zeros(shape.0 + 1);
            return Self::new(Array1::zeros(0), Array1::zeros(0), indptr, shape);
        }

        let nnz = rows.len();
        let mut all_data: Vec<(usize, usize, T)> = Vec::with_capacity(nnz);

        for i in 0..nnz {
            if rows[i] >= shape.0 || cols[i] >= shape.1 {
                return Err(SparseError::IndexOutOfBounds {
                    index: (rows[i], cols[i]),
                    shape,
                });
            }
            all_data.push((rows[i], cols[i], data[i]));
        }

        if !sorted {
            all_data.sort_by_key(|&(row, col, _)| (row, col));
        }

        // Count elements per row
        let mut row_counts = vec![0; shape.0];
        for &(row, _, _) in &all_data {
            row_counts[row] += 1;
        }

        // Create indptr
        let mut indptr = Vec::with_capacity(shape.0 + 1);
        indptr.push(0);
        let mut cumsum = 0;
        for &count in &row_counts {
            cumsum += count;
            indptr.push(cumsum);
        }

        // Create indices and data arrays
        let mut indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for (_, col, val) in all_data {
            indices.push(col);
            values.push(val);
        }

        Self::new(
            Array1::from_vec(values),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            shape,
        )
    }

    /// Checks if column indices are sorted for each row
    fn check_sorted_indices(indices: &Array1<usize>, indptr: &Array1<usize>) -> bool {
        for row in 0..indptr.len() - 1 {
            let start = indptr[row];
            let end = indptr[row + 1];

            for i in start..end.saturating_sub(1) {
                if i + 1 < indices.len() && indices[i] > indices[i + 1] {
                    return false;
                }
            }
        }
        true
    }

    /// Get the raw data array
    pub fn get_data(&self) -> &Array1<T> {
        &self.data
    }

    /// Get the raw indices array
    pub fn get_indices(&self) -> &Array1<usize> {
        &self.indices
    }

    /// Get the raw indptr array
    pub fn get_indptr(&self) -> &Array1<usize> {
        &self.indptr
    }

    /// Get the number of rows
    pub fn nrows(&self) -> usize {
        self.shape.0
    }

    /// Get the number of columns  
    pub fn ncols(&self) -> usize {
        self.shape.1
    }

    /// Get the shape (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
}

impl<T> SparseArray<T> for CsrArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static,
{
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn nnz(&self) -> usize {
        self.data.len()
    }

    fn dtype(&self) -> &str {
        "float" // Placeholder, ideally we would return the actual type
    }

    fn to_array(&self) -> Array2<T> {
        let (rows, cols) = self.shape;
        let mut result = Array2::zeros((rows, cols));

        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            for i in start..end {
                let col = self.indices[i];
                result[[row, col]] = self.data[i];
            }
        }

        result
    }

    fn toarray(&self) -> Array2<T> {
        self.to_array()
    }

    fn to_coo(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // This would convert to COO format
        // For now we just return self
        Ok(Box::new(self.clone()))
    }

    fn to_csr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        Ok(Box::new(self.clone()))
    }

    fn to_csc(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // This would convert to CSC format
        // For now we just return self
        Ok(Box::new(self.clone()))
    }

    fn to_dok(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // This would convert to DOK format
        // For now we just return self
        Ok(Box::new(self.clone()))
    }

    fn to_lil(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // This would convert to LIL format
        // For now we just return self
        Ok(Box::new(self.clone()))
    }

    fn to_dia(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // This would convert to DIA format
        // For now we just return self
        Ok(Box::new(self.clone()))
    }

    fn to_bsr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // This would convert to BSR format
        // For now we just return self
        Ok(Box::new(self.clone()))
    }

    fn add(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Placeholder implementation - a full implementation would handle direct
        // sparse matrix addition more efficiently
        let self_array = self.to_array();
        let other_array = other.to_array();

        if self.shape() != other.shape() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape().0,
                found: other.shape().0,
            });
        }

        let result = &self_array + &other_array;

        // Convert back to CSR
        let (rows, cols) = self.shape();
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for row in 0..rows {
            for col in 0..cols {
                let val = result[[row, col]];
                if !val.is_zero() {
                    data.push(val);
                    indices.push(col);
                }
            }
            indptr.push(data.len());
        }

        CsrArray::new(
            Array1::from_vec(data),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            self.shape(),
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn sub(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Similar to add, this is a placeholder
        let self_array = self.to_array();
        let other_array = other.to_array();

        if self.shape() != other.shape() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape().0,
                found: other.shape().0,
            });
        }

        let result = &self_array - &other_array;

        // Convert back to CSR
        let (rows, cols) = self.shape();
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for row in 0..rows {
            for col in 0..cols {
                let val = result[[row, col]];
                if !val.is_zero() {
                    data.push(val);
                    indices.push(col);
                }
            }
            indptr.push(data.len());
        }

        CsrArray::new(
            Array1::from_vec(data),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            self.shape(),
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn mul(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // This is element-wise multiplication (Hadamard product)
        // In the sparse array API, * is element-wise, not matrix multiplication
        let self_array = self.to_array();
        let other_array = other.to_array();

        if self.shape() != other.shape() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape().0,
                found: other.shape().0,
            });
        }

        let result = &self_array * &other_array;

        // Convert back to CSR
        let (rows, cols) = self.shape();
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for row in 0..rows {
            for col in 0..cols {
                let val = result[[row, col]];
                if !val.is_zero() {
                    data.push(val);
                    indices.push(col);
                }
            }
            indptr.push(data.len());
        }

        CsrArray::new(
            Array1::from_vec(data),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            self.shape(),
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn div(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Element-wise division
        let self_array = self.to_array();
        let other_array = other.to_array();

        if self.shape() != other.shape() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape().0,
                found: other.shape().0,
            });
        }

        let result = &self_array / &other_array;

        // Convert back to CSR
        let (rows, cols) = self.shape();
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for row in 0..rows {
            for col in 0..cols {
                let val = result[[row, col]];
                if !val.is_zero() {
                    data.push(val);
                    indices.push(col);
                }
            }
            indptr.push(data.len());
        }

        CsrArray::new(
            Array1::from_vec(data),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            self.shape(),
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn dot(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Matrix multiplication
        // This is a placeholder; a full implementation would be optimized for sparse arrays
        let (m, n) = self.shape();
        let (p, q) = other.shape();

        if n != p {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: p,
            });
        }

        let mut result = Array2::zeros((m, q));
        let other_array = other.to_array();

        for row in 0..m {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            for j in 0..q {
                let mut sum = T::zero();
                for idx in start..end {
                    let col = self.indices[idx];
                    sum = sum + self.data[idx] * other_array[[col, j]];
                }
                if !sum.is_zero() {
                    result[[row, j]] = sum;
                }
            }
        }

        // Convert result back to CSR
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for row in 0..m {
            for col in 0..q {
                let val = result[[row, col]];
                if !val.is_zero() {
                    data.push(val);
                    indices.push(col);
                }
            }
            indptr.push(data.len());
        }

        CsrArray::new(
            Array1::from_vec(data),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            (m, q),
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn dot_vector(&self, other: &ArrayView1<T>) -> SparseResult<Array1<T>> {
        let (m, n) = self.shape();
        if n != other.len() {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: other.len(),
            });
        }

        let mut result = Array1::zeros(m);

        for row in 0..m {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            let mut sum = T::zero();
            for idx in start..end {
                let col = self.indices[idx];
                sum = sum + self.data[idx] * other[col];
            }
            result[row] = sum;
        }

        Ok(result)
    }

    fn transpose(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Transpose is non-trivial for CSR format
        // A full implementation would convert to CSC format or implement
        // an efficient algorithm
        let (rows, cols) = self.shape();
        let mut row_indices = Vec::with_capacity(self.nnz());
        let mut col_indices = Vec::with_capacity(self.nnz());
        let mut values = Vec::with_capacity(self.nnz());

        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            for idx in start..end {
                let col = self.indices[idx];
                row_indices.push(col); // Note: rows and cols are swapped for transposition
                col_indices.push(row);
                values.push(self.data[idx]);
            }
        }

        // We need to create CSR from this "COO" representation
        CsrArray::from_triplets(
            &row_indices,
            &col_indices,
            &values,
            (cols, rows), // Swapped dimensions
            false,
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn copy(&self) -> Box<dyn SparseArray<T>> {
        Box::new(self.clone())
    }

    fn get(&self, i: usize, j: usize) -> T {
        if i >= self.shape.0 || j >= self.shape.1 {
            return T::zero();
        }

        let start = self.indptr[i];
        let end = self.indptr[i + 1];

        for idx in start..end {
            if self.indices[idx] == j {
                return self.data[idx];
            }
            // If indices are sorted, we can break early
            if self.has_sorted_indices && self.indices[idx] > j {
                break;
            }
        }

        T::zero()
    }

    fn set(&mut self, i: usize, j: usize, value: T) -> SparseResult<()> {
        // Setting elements in CSR format is non-trivial
        // This is a placeholder implementation that doesn't actually modify the structure
        if i >= self.shape.0 || j >= self.shape.1 {
            return Err(SparseError::IndexOutOfBounds {
                index: (i, j),
                shape: self.shape,
            });
        }

        let start = self.indptr[i];
        let end = self.indptr[i + 1];

        // Try to find existing element
        for idx in start..end {
            if self.indices[idx] == j {
                // Update value
                self.data[idx] = value;
                return Ok(());
            }
            // If indices are sorted, we can insert at the right position
            if self.has_sorted_indices && self.indices[idx] > j {
                // Insert here - this would require restructuring indices and data
                // Not implemented in this placeholder
                return Err(SparseError::NotImplemented(
                    "Inserting new elements in CSR format".to_string(),
                ));
            }
        }

        // Element not found, would need to insert
        // This would require restructuring indices and data
        Err(SparseError::NotImplemented(
            "Inserting new elements in CSR format".to_string(),
        ))
    }

    fn eliminate_zeros(&mut self) {
        // Find all non-zero entries
        let mut new_data = Vec::new();
        let mut new_indices = Vec::new();
        let mut new_indptr = vec![0];

        let (rows, _) = self.shape();

        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            for idx in start..end {
                if !self.data[idx].is_zero() {
                    new_data.push(self.data[idx]);
                    new_indices.push(self.indices[idx]);
                }
            }
            new_indptr.push(new_data.len());
        }

        // Replace data with filtered data
        self.data = Array1::from_vec(new_data);
        self.indices = Array1::from_vec(new_indices);
        self.indptr = Array1::from_vec(new_indptr);
    }

    fn sort_indices(&mut self) {
        if self.has_sorted_indices {
            return;
        }

        let (rows, _) = self.shape();

        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            if start == end {
                continue;
            }

            // Extract the non-zero elements for this row
            let mut row_data = Vec::with_capacity(end - start);
            for idx in start..end {
                row_data.push((self.indices[idx], self.data[idx]));
            }

            // Sort by column index
            row_data.sort_by_key(|&(col, _)| col);

            // Put the sorted data back
            for (i, (col, val)) in row_data.into_iter().enumerate() {
                self.indices[start + i] = col;
                self.data[start + i] = val;
            }
        }

        self.has_sorted_indices = true;
    }

    fn sorted_indices(&self) -> Box<dyn SparseArray<T>> {
        if self.has_sorted_indices {
            return Box::new(self.clone());
        }

        let mut sorted = self.clone();
        sorted.sort_indices();
        Box::new(sorted)
    }

    fn has_sorted_indices(&self) -> bool {
        self.has_sorted_indices
    }

    fn sum(&self, axis: Option<usize>) -> SparseResult<SparseSum<T>> {
        match axis {
            None => {
                // Sum all elements
                let mut sum = T::zero();
                for &val in self.data.iter() {
                    sum = sum + val;
                }
                Ok(SparseSum::Scalar(sum))
            }
            Some(0) => {
                // Sum along rows (result is a row vector)
                let (_, cols) = self.shape();
                let mut result = vec![T::zero(); cols];

                for row in 0..self.shape.0 {
                    let start = self.indptr[row];
                    let end = self.indptr[row + 1];

                    for idx in start..end {
                        let col = self.indices[idx];
                        result[col] = result[col] + self.data[idx];
                    }
                }

                // Convert to CSR format
                let mut data = Vec::new();
                let mut indices = Vec::new();
                let mut indptr = vec![0];

                for (col, &val) in result.iter().enumerate() {
                    if !val.is_zero() {
                        data.push(val);
                        indices.push(col);
                    }
                }
                indptr.push(data.len());

                let result_array = CsrArray::new(
                    Array1::from_vec(data),
                    Array1::from_vec(indices),
                    Array1::from_vec(indptr),
                    (1, cols),
                )?;

                Ok(SparseSum::SparseArray(Box::new(result_array)))
            }
            Some(1) => {
                // Sum along columns (result is a column vector)
                let mut result = Vec::with_capacity(self.shape.0);

                for row in 0..self.shape.0 {
                    let start = self.indptr[row];
                    let end = self.indptr[row + 1];

                    let mut row_sum = T::zero();
                    for idx in start..end {
                        row_sum = row_sum + self.data[idx];
                    }
                    result.push(row_sum);
                }

                // Convert to CSR format
                let mut data = Vec::new();
                let mut indices = Vec::new();
                let mut indptr = vec![0];

                for &val in result.iter() {
                    if !val.is_zero() {
                        data.push(val);
                        indices.push(0);
                        indptr.push(data.len());
                    } else {
                        indptr.push(data.len());
                    }
                }

                let result_array = CsrArray::new(
                    Array1::from_vec(data),
                    Array1::from_vec(indices),
                    Array1::from_vec(indptr),
                    (self.shape.0, 1),
                )?;

                Ok(SparseSum::SparseArray(Box::new(result_array)))
            }
            _ => Err(SparseError::InvalidAxis),
        }
    }

    fn max(&self) -> T {
        if self.data.is_empty() {
            return T::neg_infinity();
        }

        let mut max_val = self.data[0];
        for &val in self.data.iter().skip(1) {
            if val > max_val {
                max_val = val;
            }
        }

        // Check if max_val is less than zero, as zeros aren't explicitly stored
        if max_val < T::zero() && self.nnz() < self.shape.0 * self.shape.1 {
            max_val = T::zero();
        }

        max_val
    }

    fn min(&self) -> T {
        if self.data.is_empty() {
            return T::infinity();
        }

        let mut min_val = self.data[0];
        for &val in self.data.iter().skip(1) {
            if val < min_val {
                min_val = val;
            }
        }

        // Check if min_val is greater than zero, as zeros aren't explicitly stored
        if min_val > T::zero() && self.nnz() < self.shape.0 * self.shape.1 {
            min_val = T::zero();
        }

        min_val
    }

    fn find(&self) -> (Array1<usize>, Array1<usize>, Array1<T>) {
        let nnz = self.nnz();
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for row in 0..self.shape.0 {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            for idx in start..end {
                let col = self.indices[idx];
                rows.push(row);
                cols.push(col);
                values.push(self.data[idx]);
            }
        }

        (
            Array1::from_vec(rows),
            Array1::from_vec(cols),
            Array1::from_vec(values),
        )
    }

    fn slice(
        &self,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (start_row, end_row) = row_range;
        let (start_col, end_col) = col_range;

        if start_row >= self.shape.0
            || end_row > self.shape.0
            || start_col >= self.shape.1
            || end_col > self.shape.1
        {
            return Err(SparseError::InvalidSliceRange);
        }

        if start_row >= end_row || start_col >= end_col {
            return Err(SparseError::InvalidSliceRange);
        }

        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for row in start_row..end_row {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];

            for idx in start..end {
                let col = self.indices[idx];
                if col >= start_col && col < end_col {
                    data.push(self.data[idx]);
                    indices.push(col - start_col);
                }
            }
            indptr.push(data.len());
        }

        CsrArray::new(
            Array1::from_vec(data),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            (end_row - start_row, end_col - start_col),
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<T> fmt::Debug for CsrArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CsrArray<{}x{}, nnz={}>",
            self.shape.0,
            self.shape.1,
            self.nnz()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_csr_array_construction() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let indices = Array1::from_vec(vec![0, 2, 1, 0, 2]);
        let indptr = Array1::from_vec(vec![0, 2, 3, 5]);
        let shape = (3, 3);

        let csr = CsrArray::new(data, indices, indptr, shape).unwrap();

        assert_eq!(csr.shape(), (3, 3));
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.get(0, 0), 1.0);
        assert_eq!(csr.get(0, 2), 2.0);
        assert_eq!(csr.get(1, 1), 3.0);
        assert_eq!(csr.get(2, 0), 4.0);
        assert_eq!(csr.get(2, 2), 5.0);
        assert_eq!(csr.get(0, 1), 0.0);
    }

    #[test]
    fn test_csr_from_triplets() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let csr = CsrArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();

        assert_eq!(csr.shape(), (3, 3));
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.get(0, 0), 1.0);
        assert_eq!(csr.get(0, 2), 2.0);
        assert_eq!(csr.get(1, 1), 3.0);
        assert_eq!(csr.get(2, 0), 4.0);
        assert_eq!(csr.get(2, 2), 5.0);
        assert_eq!(csr.get(0, 1), 0.0);
    }

    #[test]
    fn test_csr_array_to_array() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let indices = Array1::from_vec(vec![0, 2, 1, 0, 2]);
        let indptr = Array1::from_vec(vec![0, 2, 3, 5]);
        let shape = (3, 3);

        let csr = CsrArray::new(data, indices, indptr, shape).unwrap();
        let dense = csr.to_array();

        assert_eq!(dense.shape(), &[3, 3]);
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[0, 1]], 0.0);
        assert_eq!(dense[[0, 2]], 2.0);
        assert_eq!(dense[[1, 0]], 0.0);
        assert_eq!(dense[[1, 1]], 3.0);
        assert_eq!(dense[[1, 2]], 0.0);
        assert_eq!(dense[[2, 0]], 4.0);
        assert_eq!(dense[[2, 1]], 0.0);
        assert_eq!(dense[[2, 2]], 5.0);
    }

    #[test]
    fn test_csr_array_dot_vector() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let indices = Array1::from_vec(vec![0, 2, 1, 0, 2]);
        let indptr = Array1::from_vec(vec![0, 2, 3, 5]);
        let shape = (3, 3);

        let csr = CsrArray::new(data, indices, indptr, shape).unwrap();
        let vec = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = csr.dot_vector(&vec.view()).unwrap();

        // Expected: [1*1 + 0*2 + 2*3, 0*1 + 3*2 + 0*3, 4*1 + 0*2 + 5*3] = [7, 6, 19]
        assert_eq!(result.len(), 3);
        assert_relative_eq!(result[0], 7.0);
        assert_relative_eq!(result[1], 6.0);
        assert_relative_eq!(result[2], 19.0);
    }

    #[test]
    fn test_csr_array_sum() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let indices = Array1::from_vec(vec![0, 2, 1, 0, 2]);
        let indptr = Array1::from_vec(vec![0, 2, 3, 5]);
        let shape = (3, 3);

        let csr = CsrArray::new(data, indices, indptr, shape).unwrap();

        // Sum all elements
        if let SparseSum::Scalar(sum) = csr.sum(None).unwrap() {
            assert_relative_eq!(sum, 15.0);
        } else {
            panic!("Expected scalar sum");
        }

        // Sum along rows
        if let SparseSum::SparseArray(row_sum) = csr.sum(Some(0)).unwrap() {
            let row_sum_array = row_sum.to_array();
            assert_eq!(row_sum_array.shape(), &[1, 3]);
            assert_relative_eq!(row_sum_array[[0, 0]], 5.0);
            assert_relative_eq!(row_sum_array[[0, 1]], 3.0);
            assert_relative_eq!(row_sum_array[[0, 2]], 7.0);
        } else {
            panic!("Expected sparse array sum");
        }

        // Sum along columns
        if let SparseSum::SparseArray(col_sum) = csr.sum(Some(1)).unwrap() {
            let col_sum_array = col_sum.to_array();
            assert_eq!(col_sum_array.shape(), &[3, 1]);
            assert_relative_eq!(col_sum_array[[0, 0]], 3.0);
            assert_relative_eq!(col_sum_array[[1, 0]], 3.0);
            assert_relative_eq!(col_sum_array[[2, 0]], 9.0);
        } else {
            panic!("Expected sparse array sum");
        }
    }
}
