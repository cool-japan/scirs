// CSC Array implementation
//
// This module provides the CSC (Compressed Sparse Column) array format,
// which is efficient for column-wise operations.

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use std::fmt::{self, Debug};
use std::ops::{Add, Div, Mul, Sub};

use crate::coo_array::CooArray;
use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::{SparseArray, SparseSum};

/// CSC Array format
///
/// The CSC (Compressed Sparse Column) format stores a sparse array in three arrays:
/// - data: array of non-zero values
/// - indices: row indices of the non-zero values
/// - indptr: index pointers; for each column, points to the first non-zero element
///
/// # Notes
///
/// - Efficient for column-oriented operations
/// - Fast matrix-vector multiplications for A^T x
/// - Fast column slicing
/// - Slow row slicing
/// - Slow constructing by setting individual elements
///
#[derive(Clone)]
pub struct CscArray<T>
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
    /// Row indices of non-zero values
    indices: Array1<usize>,
    /// Column pointers (indices into data/indices for the start of each column)
    indptr: Array1<usize>,
    /// Shape of the sparse array
    shape: (usize, usize),
    /// Whether indices are sorted for each column
    has_sorted_indices: bool,
}

impl<T> CscArray<T>
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
    /// Creates a new CSC array from raw components
    ///
    /// # Arguments
    /// * `data` - Array of non-zero values
    /// * `indices` - Row indices of non-zero values
    /// * `indptr` - Index pointers for the start of each column
    /// * `shape` - Shape of the sparse array
    ///
    /// # Returns
    /// A new `CscArray`
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

        if indptr.len() != shape.1 + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "indptr length ({}) must be one more than the number of columns ({})",
                    indptr.len(),
                    shape.1
                ),
            });
        }

        if let Some(&max_idx) = indices.iter().max() {
            if max_idx >= shape.0 {
                return Err(SparseError::IndexOutOfBounds {
                    index: (max_idx, 0),
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

    /// Create a CSC array from triplet format (COO-like)
    ///
    /// # Arguments
    /// * `rows` - Row indices
    /// * `cols` - Column indices
    /// * `data` - Values
    /// * `shape` - Shape of the sparse array
    /// * `sorted` - Whether the triplets are sorted by column
    ///
    /// # Returns
    /// A new `CscArray`
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
            let indptr = Array1::zeros(shape.1 + 1);
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
            all_data.sort_by_key(|&(_, col, _)| col);
        }

        // Count elements per column
        let mut col_counts = vec![0; shape.1];
        for &(_, col, _) in &all_data {
            col_counts[col] += 1;
        }

        // Create indptr
        let mut indptr = Vec::with_capacity(shape.1 + 1);
        indptr.push(0);
        let mut cumsum = 0;
        for &count in &col_counts {
            cumsum += count;
            indptr.push(cumsum);
        }

        // Create indices and data arrays
        let mut indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for (row, _, val) in all_data {
            indices.push(row);
            values.push(val);
        }

        Self::new(
            Array1::from_vec(values),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            shape,
        )
    }

    /// Checks if row indices are sorted for each column
    fn check_sorted_indices(indices: &Array1<usize>, indptr: &Array1<usize>) -> bool {
        for col in 0..indptr.len() - 1 {
            let start = indptr[col];
            let end = indptr[col + 1];

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
}

impl<T> SparseArray<T> for CscArray<T>
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

        for col in 0..cols {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];

            for i in start..end {
                let row = self.indices[i];
                result[[row, col]] = self.data[i];
            }
        }

        result
    }

    fn toarray(&self) -> Array2<T> {
        self.to_array()
    }

    fn to_coo(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to COO format
        let nnz = self.nnz();
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for col in 0..self.shape.1 {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];

            for idx in start..end {
                row_indices.push(self.indices[idx]);
                col_indices.push(col);
                values.push(self.data[idx]);
            }
        }

        CooArray::from_triplets(&row_indices, &col_indices, &values, self.shape, false)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_csr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, we'll go via COO format
        self.to_coo()?.to_csr()
    }

    fn to_csc(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        Ok(Box::new(self.clone()))
    }

    fn to_dok(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // This would convert to DOK format
        // For now, we'll go via COO format
        self.to_coo()?.to_dok()
    }

    fn to_lil(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // This would convert to LIL format
        // For now, we'll go via COO format
        self.to_coo()?.to_lil()
    }

    fn to_dia(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // This would convert to DIA format
        // For now, we'll go via COO format
        self.to_coo()?.to_dia()
    }

    fn to_bsr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // This would convert to BSR format
        // For now, we'll go via COO format
        self.to_coo()?.to_bsr()
    }

    fn add(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert to CSR format and then add
        let self_csr = self.to_csr()?;
        self_csr.add(other)
    }

    fn sub(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert to CSR format and then subtract
        let self_csr = self.to_csr()?;
        self_csr.sub(other)
    }

    fn mul(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Element-wise multiplication (Hadamard product)
        // Convert to CSR for efficiency
        let self_csr = self.to_csr()?;
        self_csr.mul(other)
    }

    fn div(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Element-wise division
        // Convert to CSR for efficiency
        let self_csr = self.to_csr()?;
        self_csr.div(other)
    }

    fn dot(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Matrix multiplication
        // Convert to CSR for efficiency
        let self_csr = self.to_csr()?;
        self_csr.dot(other)
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

        for col in 0..n {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];

            let val = other[col];
            if !val.is_zero() {
                for idx in start..end {
                    let row = self.indices[idx];
                    result[row] = result[row] + self.data[idx] * val;
                }
            }
        }

        Ok(result)
    }

    fn transpose(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // CSC transposed is effectively CSR (swap rows/cols)
        CsrArray::new(
            self.data.clone(),
            self.indices.clone(),
            self.indptr.clone(),
            (self.shape.1, self.shape.0), // Swap dimensions
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

        let start = self.indptr[j];
        let end = self.indptr[j + 1];

        for idx in start..end {
            if self.indices[idx] == i {
                return self.data[idx];
            }

            // If indices are sorted, we can break early
            if self.has_sorted_indices && self.indices[idx] > i {
                break;
            }
        }

        T::zero()
    }

    fn set(&mut self, i: usize, j: usize, value: T) -> SparseResult<()> {
        // Setting elements in CSC format is non-trivial
        // This is a placeholder implementation that doesn't actually modify the structure
        if i >= self.shape.0 || j >= self.shape.1 {
            return Err(SparseError::IndexOutOfBounds {
                index: (i, j),
                shape: self.shape,
            });
        }

        let start = self.indptr[j];
        let end = self.indptr[j + 1];

        // Try to find existing element
        for idx in start..end {
            if self.indices[idx] == i {
                // Update value
                self.data[idx] = value;
                return Ok(());
            }

            // If indices are sorted, we can insert at the right position
            if self.has_sorted_indices && self.indices[idx] > i {
                // Insert here - this would require restructuring indices and data
                // Not implemented in this placeholder
                return Err(SparseError::NotImplemented(
                    "Inserting new elements in CSC format".to_string(),
                ));
            }
        }

        // Element not found, would need to insert
        // This would require restructuring indices and data
        Err(SparseError::NotImplemented(
            "Inserting new elements in CSC format".to_string(),
        ))
    }

    fn eliminate_zeros(&mut self) {
        // Find all non-zero entries
        let mut new_data = Vec::new();
        let mut new_indices = Vec::new();
        let mut new_indptr = vec![0];

        let (_, cols) = self.shape;

        for col in 0..cols {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];

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

        let (_, cols) = self.shape;

        for col in 0..cols {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];

            if start == end {
                continue;
            }

            // Extract the non-zero elements for this column
            let mut col_data = Vec::with_capacity(end - start);
            for idx in start..end {
                col_data.push((self.indices[idx], self.data[idx]));
            }

            // Sort by row index
            col_data.sort_by_key(|&(row, _)| row);

            // Put the sorted data back
            for (i, (row, val)) in col_data.into_iter().enumerate() {
                self.indices[start + i] = row;
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
                // For efficiency, convert to CSR and sum
                let self_csr = self.to_csr()?;
                self_csr.sum(Some(0))
            }
            Some(1) => {
                // Sum along columns (result is a column vector)
                let mut result = Vec::with_capacity(self.shape.1);

                for col in 0..self.shape.1 {
                    let start = self.indptr[col];
                    let end = self.indptr[col + 1];

                    let mut col_sum = T::zero();
                    for idx in start..end {
                        col_sum = col_sum + self.data[idx];
                    }
                    result.push(col_sum);
                }

                // Convert to COO format for the column vector
                let mut row_indices = Vec::new();
                let mut col_indices = Vec::new();
                let mut values = Vec::new();

                for (col, &val) in result.iter().enumerate() {
                    if !val.is_zero() {
                        row_indices.push(0);
                        col_indices.push(col);
                        values.push(val);
                    }
                }

                let coo = CooArray::from_triplets(
                    &row_indices,
                    &col_indices,
                    &values,
                    (1, self.shape.1),
                    true,
                )?;

                Ok(SparseSum::SparseArray(Box::new(coo)))
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

        for col in 0..self.shape.1 {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];

            for idx in start..end {
                let row = self.indices[idx];
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

        // CSC format is efficient for column slicing
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for col in start_col..end_col {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];

            for idx in start..end {
                let row = self.indices[idx];
                if row >= start_row && row < end_row {
                    data.push(self.data[idx]);
                    indices.push(row - start_row); // Adjust indices
                }
            }
            indptr.push(data.len());
        }

        CscArray::new(
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

impl<T> fmt::Debug for CscArray<T>
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
            "CscArray<{}x{}, nnz={}>",
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
    fn test_csc_array_construction() {
        let data = Array1::from_vec(vec![1.0, 4.0, 2.0, 3.0, 5.0]);
        let indices = Array1::from_vec(vec![0, 2, 0, 1, 2]);
        let indptr = Array1::from_vec(vec![0, 2, 3, 5]);
        let shape = (3, 3);

        let csc = CscArray::new(data, indices, indptr, shape).unwrap();

        assert_eq!(csc.shape(), (3, 3));
        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.get(0, 0), 1.0);
        assert_eq!(csc.get(2, 0), 4.0);
        assert_eq!(csc.get(0, 1), 2.0);
        assert_eq!(csc.get(1, 2), 3.0);
        assert_eq!(csc.get(2, 2), 5.0);
        assert_eq!(csc.get(1, 0), 0.0);
    }

    #[test]
    fn test_csc_from_triplets() {
        let rows = vec![0, 2, 0, 1, 2];
        let cols = vec![0, 0, 1, 2, 2];
        let data = vec![1.0, 4.0, 2.0, 3.0, 5.0];
        let shape = (3, 3);

        let csc = CscArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();

        assert_eq!(csc.shape(), (3, 3));
        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.get(0, 0), 1.0);
        assert_eq!(csc.get(2, 0), 4.0);
        assert_eq!(csc.get(0, 1), 2.0);
        assert_eq!(csc.get(1, 2), 3.0);
        assert_eq!(csc.get(2, 2), 5.0);
        assert_eq!(csc.get(1, 0), 0.0);
    }

    #[test]
    fn test_csc_array_to_array() {
        let rows = vec![0, 2, 0, 1, 2];
        let cols = vec![0, 0, 1, 2, 2];
        let data = vec![1.0, 4.0, 2.0, 3.0, 5.0];
        let shape = (3, 3);

        let csc = CscArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let dense = csc.to_array();

        assert_eq!(dense.shape(), &[3, 3]);
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[1, 0]], 0.0);
        assert_eq!(dense[[2, 0]], 4.0);
        assert_eq!(dense[[0, 1]], 2.0);
        assert_eq!(dense[[1, 1]], 0.0);
        assert_eq!(dense[[2, 1]], 0.0);
        assert_eq!(dense[[0, 2]], 0.0);
        assert_eq!(dense[[1, 2]], 3.0);
        assert_eq!(dense[[2, 2]], 5.0);
    }

    #[test]
    fn test_csc_to_csr_conversion() {
        let rows = vec![0, 2, 0, 1, 2];
        let cols = vec![0, 0, 1, 2, 2];
        let data = vec![1.0, 4.0, 2.0, 3.0, 5.0];
        let shape = (3, 3);

        let csc = CscArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let csr = csc.to_csr().unwrap();

        // Check that the conversion preserved values
        let csc_array = csc.to_array();
        let csr_array = csr.to_array();

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                assert_relative_eq!(csc_array[[i, j]], csr_array[[i, j]]);
            }
        }
    }

    #[test]
    fn test_csc_dot_vector() {
        let rows = vec![0, 2, 0, 1, 2];
        let cols = vec![0, 0, 1, 2, 2];
        let data = vec![1.0, 4.0, 2.0, 3.0, 5.0];
        let shape = (3, 3);

        let csc = CscArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let vec = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = csc.dot_vector(&vec.view()).unwrap();

        // Expected:
        // [0,0]*1 + [0,1]*2 + [0,2]*3 = 1*1 + 2*2 + 0*3 = 5
        // [1,0]*1 + [1,1]*2 + [1,2]*3 = 0*1 + 0*2 + 3*3 = 9
        // [2,0]*1 + [2,1]*2 + [2,2]*3 = 4*1 + 0*2 + 5*3 = 19
        assert_eq!(result.len(), 3);
        assert_relative_eq!(result[0], 5.0);
        assert_relative_eq!(result[1], 9.0);
        assert_relative_eq!(result[2], 19.0);
    }

    #[test]
    fn test_csc_transpose() {
        let rows = vec![0, 2, 0, 1, 2];
        let cols = vec![0, 0, 1, 2, 2];
        let data = vec![1.0, 4.0, 2.0, 3.0, 5.0];
        let shape = (3, 3);

        let csc = CscArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let transposed = csc.transpose().unwrap();

        // Check dimensions are swapped
        assert_eq!(transposed.shape(), (3, 3));

        // Check values are correctly transposed
        let dense = transposed.to_array();
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[0, 2]], 4.0);
        assert_eq!(dense[[1, 0]], 2.0);
        assert_eq!(dense[[2, 1]], 3.0);
        assert_eq!(dense[[2, 2]], 5.0);
    }

    #[test]
    fn test_csc_find() {
        let rows = vec![0, 2, 0, 1, 2];
        let cols = vec![0, 0, 1, 2, 2];
        let data = vec![1.0, 4.0, 2.0, 3.0, 5.0];
        let shape = (3, 3);

        let csc = CscArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let (result_rows, result_cols, result_data) = csc.find();

        // Check that the find operation returned all non-zero elements
        assert_eq!(result_rows.len(), 5);
        assert_eq!(result_cols.len(), 5);
        assert_eq!(result_data.len(), 5);

        // Create vectors of tuples to compare
        let mut original: Vec<_> = rows
            .iter()
            .zip(cols.iter())
            .zip(data.iter())
            .map(|((r, c), d)| (*r, *c, *d))
            .collect();

        let mut result: Vec<_> = result_rows
            .iter()
            .zip(result_cols.iter())
            .zip(result_data.iter())
            .map(|((r, c), d)| (*r, *c, *d))
            .collect();

        // Sort the vectors before comparing
        original.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        result.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        assert_eq!(original, result);
    }
}
