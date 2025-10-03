// COO Array implementation
//
// This module provides the COO (COOrdinate) array format,
// which is efficient for incrementally constructing a sparse array.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float;
use std::fmt::{self, Debug};
use std::ops::{Add, Div, Mul, Sub};

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::{SparseArray, SparseSum};

/// COO Array format
///
/// The COO (COOrdinate) format stores data as a triplet of arrays:
/// - row: array of row indices
/// - col: array of column indices
/// - data: array of corresponding non-zero values
///
/// # Notes
///
/// - Efficient for incrementally constructing a sparse array or matrix
/// - Allows for duplicate entries (summed when converted to other formats)
/// - Fast conversion to other formats
/// - Not efficient for arithmetic operations
/// - Not efficient for slicing operations
///
#[derive(Clone)]
pub struct CooArray<T>
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
    /// Row indices
    row: Array1<usize>,
    /// Column indices
    col: Array1<usize>,
    /// Data values
    data: Array1<T>,
    /// Shape of the array
    shape: (usize, usize),
    /// Whether entries are sorted by row
    has_canonical_format: bool,
}

impl<T> CooArray<T>
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
    /// Creates a new COO array
    ///
    /// # Arguments
    /// * `data` - Array of non-zero values
    /// * `row` - Array of row indices
    /// * `col` - Array of column indices
    /// * `shape` - Shape of the sparse array
    /// * `has_canonical_format` - Whether entries are sorted by row
    ///
    /// # Returns
    /// A new `CooArray`
    ///
    /// # Errors
    /// Returns an error if the data is not consistent
    pub fn new(
        data: Array1<T>,
        row: Array1<usize>,
        col: Array1<usize>,
        shape: (usize, usize),
        has_canonical_format: bool,
    ) -> SparseResult<Self> {
        // Validation
        if data.len() != row.len() || data.len() != col.len() {
            return Err(SparseError::InconsistentData {
                reason: "data, row, and col must have the same length".to_string(),
            });
        }

        if let Some(&max_row) = row.iter().max() {
            if max_row >= shape.0 {
                return Err(SparseError::IndexOutOfBounds {
                    index: (max_row, 0),
                    shape,
                });
            }
        }

        if let Some(&max_col) = col.iter().max() {
            if max_col >= shape.1 {
                return Err(SparseError::IndexOutOfBounds {
                    index: (0, max_col),
                    shape,
                });
            }
        }

        Ok(Self {
            data,
            row,
            col,
            shape,
            has_canonical_format,
        })
    }

    /// Create a COO array from (row, col, data) triplets
    ///
    /// # Arguments
    /// * `row` - Row indices
    /// * `col` - Column indices
    /// * `data` - Values
    /// * `shape` - Shape of the sparse array
    /// * `sorted` - Whether the triplets are already sorted
    ///
    /// # Returns
    /// A new `CooArray`
    ///
    /// # Errors
    /// Returns an error if the data is not consistent
    pub fn from_triplets(
        row: &[usize],
        col: &[usize],
        data: &[T],
        shape: (usize, usize),
        sorted: bool,
    ) -> SparseResult<Self> {
        let row_array = Array1::from_vec(row.to_vec());
        let col_array = Array1::from_vec(col.to_vec());
        let data_array = Array1::from_vec(data.to_vec());

        Self::new(data_array, row_array, col_array, shape, sorted)
    }

    /// Get the rows array
    pub fn get_rows(&self) -> &Array1<usize> {
        &self.row
    }

    /// Get the cols array
    pub fn get_cols(&self) -> &Array1<usize> {
        &self.col
    }

    /// Get the data array
    pub fn get_data(&self) -> &Array1<T> {
        &self.data
    }

    /// Put the array in canonical format (sort by row index, then column index)
    pub fn canonical_format(&mut self) {
        if self.has_canonical_format {
            return;
        }

        let n = self.data.len();
        let mut triplets: Vec<(usize, usize, T)> = Vec::with_capacity(n);

        for i in 0..n {
            triplets.push((self.row[i], self.col[i], self.data[i]));
        }

        triplets.sort_by(|&(r1, c1_, _), &(r2, c2_, _)| (r1, c1_).cmp(&(r2, c2_)));

        for (i, &(r, c, v)) in triplets.iter().enumerate() {
            self.row[i] = r;
            self.col[i] = c;
            self.data[i] = v;
        }

        self.has_canonical_format = true;
    }

    /// Converts to a COO with summed duplicate entries
    pub fn sum_duplicates(&mut self) {
        self.canonical_format();

        let n = self.data.len();
        if n == 0 {
            return;
        }

        let mut new_data = Vec::new();
        let mut new_row = Vec::new();
        let mut new_col = Vec::new();

        let mut curr_row = self.row[0];
        let mut curr_col = self.col[0];
        let mut curr_sum = self.data[0];

        for i in 1..n {
            if self.row[i] == curr_row && self.col[i] == curr_col {
                curr_sum = curr_sum + self.data[i];
            } else {
                if !curr_sum.is_zero() {
                    new_data.push(curr_sum);
                    new_row.push(curr_row);
                    new_col.push(curr_col);
                }
                curr_row = self.row[i];
                curr_col = self.col[i];
                curr_sum = self.data[i];
            }
        }

        // Add the last element
        if !curr_sum.is_zero() {
            new_data.push(curr_sum);
            new_row.push(curr_row);
            new_col.push(curr_col);
        }

        self.data = Array1::from_vec(new_data);
        self.row = Array1::from_vec(new_row);
        self.col = Array1::from_vec(new_col);
    }
}

impl<T> SparseArray<T> for CooArray<T>
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
        "float" // Placeholder
    }

    fn to_array(&self) -> Array2<T> {
        let (rows, cols) = self.shape;
        let mut result = Array2::zeros((rows, cols));

        for i in 0..self.data.len() {
            let r = self.row[i];
            let c = self.col[i];
            result[[r, c]] = result[[r, c]] + self.data[i]; // Sum duplicates
        }

        result
    }

    fn toarray(&self) -> Array2<T> {
        self.to_array()
    }

    fn to_coo(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Create a copy with summed duplicates
        let mut new_coo = self.clone();
        new_coo.sum_duplicates();
        Ok(Box::new(new_coo))
    }

    fn to_csr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR format
        let mut data_vec = self.data.to_vec();
        let mut row_vec = self.row.to_vec();
        let mut col_vec = self.col.to_vec();

        // Sort by row, then column
        let mut triplets: Vec<(usize, usize, T)> = Vec::with_capacity(data_vec.len());
        for i in 0..data_vec.len() {
            triplets.push((row_vec[i], col_vec[i], data_vec[i]));
        }
        triplets.sort_by(|&(r1, c1_, _), &(r2, c2_, _)| (r1, c1_).cmp(&(r2, c2_)));

        for (i, &(r, c, v)) in triplets.iter().enumerate() {
            row_vec[i] = r;
            col_vec[i] = c;
            data_vec[i] = v;
        }

        // Convert to CSR format using CsrArray::from_triplets
        CsrArray::from_triplets(&row_vec, &col_vec, &data_vec, self.shape, true)
            .map(|csr| Box::new(csr) as Box<dyn SparseArray<T>>)
    }

    fn to_csc(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For now, convert to CSR and then transpose
        // In an actual implementation, this would directly convert to CSC
        let csr = self.to_csr()?;
        csr.transpose()
    }

    fn to_dok(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // In a real implementation, this would convert directly to DOK format
        Ok(Box::new(self.clone()))
    }

    fn to_lil(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // In a real implementation, this would convert directly to LIL format
        Ok(Box::new(self.clone()))
    }

    fn to_dia(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // In a real implementation, this would convert directly to DIA format
        Ok(Box::new(self.clone()))
    }

    fn to_bsr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // In a real implementation, this would convert directly to BSR format
        Ok(Box::new(self.clone()))
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
        // For efficiency, convert to CSR format and then multiply
        let self_csr = self.to_csr()?;
        self_csr.mul(other)
    }

    fn div(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert to CSR format and then divide
        let self_csr = self.to_csr()?;
        self_csr.div(other)
    }

    fn dot(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert to CSR format and then do matrix multiplication
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

        for i in 0..self.data.len() {
            let row = self.row[i];
            let col = self.col[i];
            result[row] = result[row] + self.data[i] * other[col];
        }

        Ok(result)
    }

    fn transpose(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Swap row and column arrays
        CooArray::new(
            self.data.clone(),
            self.col.clone(),             // Note the swap
            self.row.clone(),             // Note the swap
            (self.shape.1, self.shape.0), // Swap shape dimensions
            self.has_canonical_format,
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

        let mut sum = T::zero();
        for idx in 0..self.data.len() {
            if self.row[idx] == i && self.col[idx] == j {
                sum = sum + self.data[idx];
            }
        }

        sum
    }

    fn set(&mut self, i: usize, j: usize, value: T) -> SparseResult<()> {
        if i >= self.shape.0 || j >= self.shape.1 {
            return Err(SparseError::IndexOutOfBounds {
                index: (i, j),
                shape: self.shape,
            });
        }

        if value.is_zero() {
            // Remove existing entries at (i, j)
            let mut new_data = Vec::new();
            let mut new_row = Vec::new();
            let mut new_col = Vec::new();

            for idx in 0..self.data.len() {
                if !(self.row[idx] == i && self.col[idx] == j) {
                    new_data.push(self.data[idx]);
                    new_row.push(self.row[idx]);
                    new_col.push(self.col[idx]);
                }
            }

            self.data = Array1::from_vec(new_data);
            self.row = Array1::from_vec(new_row);
            self.col = Array1::from_vec(new_col);
        } else {
            // First remove any existing entries
            self.set(i, j, T::zero())?;

            // Then add the new value
            let mut new_data = self.data.to_vec();
            let mut new_row = self.row.to_vec();
            let mut new_col = self.col.to_vec();

            new_data.push(value);
            new_row.push(i);
            new_col.push(j);

            self.data = Array1::from_vec(new_data);
            self.row = Array1::from_vec(new_row);
            self.col = Array1::from_vec(new_col);

            // No longer in canonical format
            self.has_canonical_format = false;
        }

        Ok(())
    }

    fn eliminate_zeros(&mut self) {
        let mut new_data = Vec::new();
        let mut new_row = Vec::new();
        let mut new_col = Vec::new();

        for i in 0..self.data.len() {
            if !self.data[i].is_zero() {
                new_data.push(self.data[i]);
                new_row.push(self.row[i]);
                new_col.push(self.col[i]);
            }
        }

        self.data = Array1::from_vec(new_data);
        self.row = Array1::from_vec(new_row);
        self.col = Array1::from_vec(new_col);
    }

    fn sort_indices(&mut self) {
        self.canonical_format();
    }

    fn sorted_indices(&self) -> Box<dyn SparseArray<T>> {
        if self.has_canonical_format {
            return Box::new(self.clone());
        }

        let mut sorted = self.clone();
        sorted.canonical_format();
        Box::new(sorted)
    }

    fn has_sorted_indices(&self) -> bool {
        self.has_canonical_format
    }

    fn sum(&self, axis: Option<usize>) -> SparseResult<SparseSum<T>> {
        // For efficiency, convert to CSR format and then sum
        let self_csr = self.to_csr()?;
        self_csr.sum(axis)
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
        // Return copies of the row, col, and data arrays
        let data_vec = self.data.to_vec();
        let row_vec = self.row.to_vec();
        let col_vec = self.col.to_vec();

        // If there are duplicate entries, sum them
        if self.has_canonical_format {
            // We can use a more efficient algorithm if already sorted
            (self.row.clone(), self.col.clone(), self.data.clone())
        } else {
            // Convert to canonical form with summed duplicates
            let mut triplets: Vec<(usize, usize, T)> = Vec::with_capacity(data_vec.len());
            for i in 0..data_vec.len() {
                triplets.push((row_vec[i], col_vec[i], data_vec[i]));
            }
            triplets.sort_by(|&(r1, c1_, _), &(r2, c2_, _)| (r1, c1_).cmp(&(r2, c2_)));

            let mut result_row = Vec::new();
            let mut result_col = Vec::new();
            let mut result_data = Vec::new();

            if !triplets.is_empty() {
                let mut curr_row = triplets[0].0;
                let mut curr_col = triplets[0].1;
                let mut curr_sum = triplets[0].2;

                for &(r, c, v) in triplets.iter().skip(1) {
                    if r == curr_row && c == curr_col {
                        curr_sum = curr_sum + v;
                    } else {
                        if !curr_sum.is_zero() {
                            result_row.push(curr_row);
                            result_col.push(curr_col);
                            result_data.push(curr_sum);
                        }
                        curr_row = r;
                        curr_col = c;
                        curr_sum = v;
                    }
                }

                // Add the last element
                if !curr_sum.is_zero() {
                    result_row.push(curr_row);
                    result_col.push(curr_col);
                    result_data.push(curr_sum);
                }
            }

            (
                Array1::from_vec(result_row),
                Array1::from_vec(result_col),
                Array1::from_vec(result_data),
            )
        }
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

        let mut new_data = Vec::new();
        let mut new_row = Vec::new();
        let mut new_col = Vec::new();

        for i in 0..self.data.len() {
            let r = self.row[i];
            let c = self.col[i];

            if r >= start_row && r < end_row && c >= start_col && c < end_col {
                new_data.push(self.data[i]);
                new_row.push(r - start_row); // Adjust indices
                new_col.push(c - start_col); // Adjust indices
            }
        }

        CooArray::new(
            Array1::from_vec(new_data),
            Array1::from_vec(new_row),
            Array1::from_vec(new_col),
            (end_row - start_row, end_col - start_col),
            false,
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<T> fmt::Debug for CooArray<T>
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
            "CooArray<{}x{}, nnz={}>",
            self.shape.0,
            self.shape.1,
            self.nnz()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_array_construction() {
        let row = Array1::from_vec(vec![0, 0, 1, 2, 2]);
        let col = Array1::from_vec(vec![0, 2, 1, 0, 2]);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let shape = (3, 3);

        let coo = CooArray::new(data, row, col, shape, false).unwrap();

        assert_eq!(coo.shape(), (3, 3));
        assert_eq!(coo.nnz(), 5);
        assert_eq!(coo.get(0, 0), 1.0);
        assert_eq!(coo.get(0, 2), 2.0);
        assert_eq!(coo.get(1, 1), 3.0);
        assert_eq!(coo.get(2, 0), 4.0);
        assert_eq!(coo.get(2, 2), 5.0);
        assert_eq!(coo.get(0, 1), 0.0);
    }

    #[test]
    fn test_coo_from_triplets() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let coo = CooArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();

        assert_eq!(coo.shape(), (3, 3));
        assert_eq!(coo.nnz(), 5);
        assert_eq!(coo.get(0, 0), 1.0);
        assert_eq!(coo.get(0, 2), 2.0);
        assert_eq!(coo.get(1, 1), 3.0);
        assert_eq!(coo.get(2, 0), 4.0);
        assert_eq!(coo.get(2, 2), 5.0);
        assert_eq!(coo.get(0, 1), 0.0);
    }

    #[test]
    fn test_coo_array_to_array() {
        let row = Array1::from_vec(vec![0, 0, 1, 2, 2]);
        let col = Array1::from_vec(vec![0, 2, 1, 0, 2]);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let shape = (3, 3);

        let coo = CooArray::new(data, row, col, shape, false).unwrap();
        let dense = coo.to_array();

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
    fn test_coo_array_duplicate_entries() {
        let row = Array1::from_vec(vec![0, 0, 0, 1, 1]);
        let col = Array1::from_vec(vec![0, 0, 1, 0, 0]);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let shape = (2, 2);

        let mut coo = CooArray::new(data, row, col, shape, false).unwrap();

        // Test summing duplicates
        coo.sum_duplicates();

        // Should now have only 3 entries
        assert_eq!(coo.nnz(), 3);
        assert_eq!(coo.get(0, 0), 3.0); // 1.0 + 2.0
        assert_eq!(coo.get(0, 1), 3.0);
        assert_eq!(coo.get(1, 0), 9.0); // 4.0 + 5.0
    }

    #[test]
    fn test_coo_set_get() {
        let row = Array1::from_vec(vec![0, 1]);
        let col = Array1::from_vec(vec![0, 1]);
        let data = Array1::from_vec(vec![1.0, 2.0]);
        let shape = (2, 2);

        let mut coo = CooArray::new(data, row, col, shape, false).unwrap();

        // Set a new value
        coo.set(0, 1, 3.0).unwrap();
        assert_eq!(coo.get(0, 1), 3.0);

        // Update an existing value
        coo.set(0, 0, 4.0).unwrap();
        assert_eq!(coo.get(0, 0), 4.0);

        // Set to zero should remove the entry
        coo.set(0, 0, 0.0).unwrap();
        assert_eq!(coo.get(0, 0), 0.0);

        // nnz should be 2 now (2.0 at (1,1) and 3.0 at (0,1))
        assert_eq!(coo.nnz(), 2);
    }

    #[test]
    fn test_coo_canonical_format() {
        let row = Array1::from_vec(vec![1, 0, 2, 0]);
        let col = Array1::from_vec(vec![1, 0, 2, 2]);
        let data = Array1::from_vec(vec![3.0, 1.0, 5.0, 2.0]);
        let shape = (3, 3);

        let mut coo = CooArray::new(data, row, col, shape, false).unwrap();

        // Not in canonical format
        assert!(!coo.has_canonical_format);

        // Sort to canonical format
        coo.canonical_format();

        // Now in canonical format
        assert!(coo.has_canonical_format);

        // Check order: (0,0), (0,2), (1,1), (2,2)
        assert_eq!(coo.row[0], 0);
        assert_eq!(coo.col[0], 0);
        assert_eq!(coo.data[0], 1.0);

        assert_eq!(coo.row[1], 0);
        assert_eq!(coo.col[1], 2);
        assert_eq!(coo.data[1], 2.0);

        assert_eq!(coo.row[2], 1);
        assert_eq!(coo.col[2], 1);
        assert_eq!(coo.data[2], 3.0);

        assert_eq!(coo.row[3], 2);
        assert_eq!(coo.col[3], 2);
        assert_eq!(coo.data[3], 5.0);
    }

    #[test]
    fn test_coo_to_csr() {
        let row = Array1::from_vec(vec![0, 0, 1, 2, 2]);
        let col = Array1::from_vec(vec![0, 2, 1, 0, 2]);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let shape = (3, 3);

        let coo = CooArray::new(data, row, col, shape, false).unwrap();

        // Convert to CSR
        let csr = coo.to_csr().unwrap();

        // Check values
        let dense = csr.to_array();
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[0, 2]], 2.0);
        assert_eq!(dense[[1, 1]], 3.0);
        assert_eq!(dense[[2, 0]], 4.0);
        assert_eq!(dense[[2, 2]], 5.0);
    }

    #[test]
    fn test_coo_transpose() {
        let row = Array1::from_vec(vec![0, 0, 1, 2, 2]);
        let col = Array1::from_vec(vec![0, 2, 1, 0, 2]);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let shape = (3, 3);

        let coo = CooArray::new(data, row, col, shape, false).unwrap();

        // Transpose
        let transposed = coo.transpose().unwrap();

        // Check shape
        assert_eq!(transposed.shape(), (3, 3));

        // Check values
        let dense = transposed.to_array();
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[2, 0]], 2.0);
        assert_eq!(dense[[1, 1]], 3.0);
        assert_eq!(dense[[0, 2]], 4.0);
        assert_eq!(dense[[2, 2]], 5.0);
    }
}
