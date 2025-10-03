// LIL Array implementation
//
// This module provides the LIL (List of Lists) array format,
// which is efficient for incremental matrix construction and row-based operations.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float;
use std::fmt::{self, Debug};
use std::ops::{Add, Div, Mul, Sub};

use crate::coo_array::CooArray;
use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::{SparseArray, SparseSum};

/// LIL Array format
///
/// The LIL (List of Lists) format stores data as a list of lists:
/// - data: a vector of vectors, where each inner vector contains the non-zero values for a row
/// - indices: a vector of vectors, where each inner vector contains the column indices for the non-zero values
///
/// # Notes
///
/// - Efficient for incremental construction (adding values row by row)
/// - Good for row-based operations
/// - Fast conversion to CSR format
/// - Not memory-efficient for large sparse arrays
/// - Not efficient for arithmetic operations
///
#[derive(Clone)]
pub struct LilArray<T>
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
    /// Data for each row (list of lists of values)
    data: Vec<Vec<T>>,
    /// Column indices for each row
    indices: Vec<Vec<usize>>,
    /// Shape of the array
    shape: (usize, usize),
}

impl<T> LilArray<T>
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
    /// Creates a new LIL array
    ///
    /// # Arguments
    /// * `shape` - Shape of the sparse array (rows, cols)
    ///
    /// # Returns
    /// A new empty `LilArray`
    pub fn new(shape: (usize, usize)) -> Self {
        let (rows, cols) = shape;
        let data = vec![Vec::new(); rows];
        let indices = vec![Vec::new(); rows];

        Self {
            data,
            indices,
            shape,
        }
    }

    /// Creates a LIL array from existing data and indices
    ///
    /// # Arguments
    /// * `data` - List of lists containing the non-zero values for each row
    /// * `indices` - List of lists containing the column indices for the non-zero values
    /// * `shape` - Shape of the sparse array
    ///
    /// # Returns
    /// A new `LilArray`
    ///
    /// # Errors
    /// Returns an error if the data and indices are not consistent or if any index is out of bounds
    pub fn from_lists(
        data: Vec<Vec<T>>,
        indices: Vec<Vec<usize>>,
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        // Validate the data
        if data.len() != shape.0 || indices.len() != shape.0 {
            return Err(SparseError::InconsistentData {
                reason: "Number of rows in data and indices must match the shape".to_string(),
            });
        }

        for (i, (row_data, row_indices)) in data.iter().zip(indices.iter()).enumerate() {
            if row_data.len() != row_indices.len() {
                return Err(SparseError::InconsistentData {
                    reason: format!("Row {i}: data and indices have different lengths"),
                });
            }

            if let Some(&max_col) = row_indices.iter().max() {
                if max_col >= shape.1 {
                    return Err(SparseError::IndexOutOfBounds {
                        index: (i, max_col),
                        shape,
                    });
                }
            }
        }

        // Create sorted copies of the data and indices
        let mut new_data = Vec::with_capacity(shape.0);
        let mut new_indices = Vec::with_capacity(shape.0);

        for (row_data, row_indices) in data.iter().zip(indices.iter()) {
            // Create sorted pairs
            let mut pairs: Vec<(usize, T)> = row_indices
                .iter()
                .copied()
                .zip(row_data.iter().copied())
                .collect();
            pairs.sort_by_key(|&(idx_, _)| idx_);

            // Extract sorted data
            let mut sorted_data = Vec::with_capacity(row_data.len());
            let mut sorted_indices = Vec::with_capacity(row_indices.len());

            for (idx, val) in pairs {
                sorted_indices.push(idx);
                sorted_data.push(val);
            }

            new_data.push(sorted_data);
            new_indices.push(sorted_indices);
        }

        Ok(Self {
            data: new_data,
            indices: new_indices,
            shape,
        })
    }

    /// Create a LIL array from (row, col, data) triplets
    ///
    /// # Arguments
    /// * `row` - Row indices
    /// * `col` - Column indices
    /// * `data` - Values
    /// * `shape` - Shape of the sparse array
    ///
    /// # Returns
    /// A new `LilArray`
    ///
    /// # Errors
    /// Returns an error if the data is not consistent
    pub fn from_triplets(
        rows: &[usize],
        cols: &[usize],
        data: &[T],
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        if rows.len() != cols.len() || rows.len() != data.len() {
            return Err(SparseError::InconsistentData {
                reason: "rows, cols, and data must have the same length".to_string(),
            });
        }

        let (num_rows, num_cols) = shape;

        // Check if any index is out of bounds
        if let Some(&max_row) = rows.iter().max() {
            if max_row >= num_rows {
                return Err(SparseError::IndexOutOfBounds {
                    index: (max_row, 0),
                    shape,
                });
            }
        }

        if let Some(&max_col) = cols.iter().max() {
            if max_col >= num_cols {
                return Err(SparseError::IndexOutOfBounds {
                    index: (0, max_col),
                    shape,
                });
            }
        }

        // Initialize the LIL array
        let mut lil = LilArray::new(shape);

        // Insert the values
        for (&row, &col, &value) in rows
            .iter()
            .zip(cols.iter())
            .zip(data.iter())
            .map(|((r, c), d)| (r, c, d))
        {
            lil.set(row, col, value)?;
        }

        Ok(lil)
    }

    /// Get a reference to the data
    pub fn get_data(&self) -> &Vec<Vec<T>> {
        &self.data
    }

    /// Get a reference to the indices
    pub fn get_indices(&self) -> &Vec<Vec<usize>> {
        &self.indices
    }

    /// Sort the indices and data in each row
    pub fn sort_indices(&mut self) {
        for row in 0..self.shape.0 {
            if !self.data[row].is_empty() {
                // Create pairs of (col_idx, value)
                let mut pairs: Vec<(usize, T)> = self.indices[row]
                    .iter()
                    .copied()
                    .zip(self.data[row].iter().copied())
                    .collect();

                // Sort by column index
                pairs.sort_by_key(|&(idx_, _)| idx_);

                // Extract sorted data
                self.indices[row].clear();
                self.data[row].clear();

                for (idx, val) in pairs {
                    self.indices[row].push(idx);
                    self.data[row].push(val);
                }
            }
        }
    }
}

impl<T> SparseArray<T> for LilArray<T>
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
        self.indices.iter().map(|row| row.len()).sum()
    }

    fn dtype(&self) -> &str {
        "float" // Placeholder
    }

    fn to_array(&self) -> Array2<T> {
        let (rows, cols) = self.shape;
        let mut result = Array2::zeros((rows, cols));

        for row in 0..rows {
            for (idx, &col) in self.indices[row].iter().enumerate() {
                result[[row, col]] = self.data[row][idx];
            }
        }

        result
    }

    fn toarray(&self) -> Array2<T> {
        self.to_array()
    }

    fn to_coo(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let nnz = self.nnz();
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for row in 0..self.shape.0 {
            for (idx, &col) in self.indices[row].iter().enumerate() {
                row_indices.push(row);
                col_indices.push(col);
                values.push(self.data[row][idx]);
            }
        }

        CooArray::from_triplets(&row_indices, &col_indices, &values, self.shape, false)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_csr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (rows, cols) = self.shape;
        let nnz = self.nnz();

        let mut data = Vec::with_capacity(nnz);
        let mut indices = Vec::with_capacity(nnz);
        let mut indptr = Vec::with_capacity(rows + 1);

        indptr.push(0);

        for row in 0..rows {
            for (idx, &col) in self.indices[row].iter().enumerate() {
                indices.push(col);
                data.push(self.data[row][idx]);
            }
            indptr.push(indptr.last().unwrap() + self.indices[row].len());
        }

        CsrArray::new(
            Array1::from_vec(data),
            Array1::from_vec(indices),
            Array1::from_vec(indptr),
            self.shape,
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_csc(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to COO first, then to CSC (for simplicity)
        self.to_coo()?.to_csc()
    }

    fn to_dok(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to COO first, then to DOK (for simplicity)
        self.to_coo()?.to_dok()
    }

    fn to_lil(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        Ok(Box::new(self.clone()))
    }

    fn to_dia(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to COO first, then to DIA (for simplicity)
        self.to_coo()?.to_dia()
    }

    fn to_bsr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to COO first, then to BSR (for simplicity)
        self.to_coo()?.to_bsr()
    }

    fn add(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert to CSR first
        let self_csr = self.to_csr()?;
        self_csr.add(other)
    }

    fn sub(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert to CSR first
        let self_csr = self.to_csr()?;
        self_csr.sub(other)
    }

    fn mul(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert to CSR first
        let self_csr = self.to_csr()?;
        self_csr.mul(other)
    }

    fn div(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert to CSR first
        let self_csr = self.to_csr()?;
        self_csr.div(other)
    }

    fn dot(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For efficiency, convert to CSR first
        let self_csr = self.to_csr()?;
        self_csr.dot(other)
    }

    fn dot_vector(&self, other: &ArrayView1<T>) -> SparseResult<Array1<T>> {
        let (rows, cols) = self.shape;
        if cols != other.len() {
            return Err(SparseError::DimensionMismatch {
                expected: cols,
                found: other.len(),
            });
        }

        let mut result = Array1::zeros(rows);

        for row in 0..rows {
            for (idx, &col) in self.indices[row].iter().enumerate() {
                result[row] = result[row] + self.data[row][idx] * other[col];
            }
        }

        Ok(result)
    }

    fn transpose(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // LIL format isn't efficient for transpose operations
        // Convert to COO first, which has a simple transpose operation
        self.to_coo()?.transpose()
    }

    fn copy(&self) -> Box<dyn SparseArray<T>> {
        Box::new(self.clone())
    }

    fn get(&self, i: usize, j: usize) -> T {
        if i >= self.shape.0 || j >= self.shape.1 {
            return T::zero();
        }

        match self.indices[i].binary_search(&j) {
            Ok(pos) => self.data[i][pos],
            Err(_) => T::zero(),
        }
    }

    fn set(&mut self, i: usize, j: usize, value: T) -> SparseResult<()> {
        if i >= self.shape.0 || j >= self.shape.1 {
            return Err(SparseError::IndexOutOfBounds {
                index: (i, j),
                shape: self.shape,
            });
        }

        match self.indices[i].binary_search(&j) {
            Ok(pos) => {
                if value.is_zero() {
                    // Remove zero value
                    self.data[i].remove(pos);
                    self.indices[i].remove(pos);
                } else {
                    // Update existing value
                    self.data[i][pos] = value;
                }
            }
            Err(pos) => {
                if !value.is_zero() {
                    // Insert new non-zero value
                    self.data[i].insert(pos, value);
                    self.indices[i].insert(pos, j);
                }
            }
        }

        Ok(())
    }

    fn eliminate_zeros(&mut self) {
        for row in 0..self.shape.0 {
            let mut new_data = Vec::new();
            let mut new_indices = Vec::new();

            for (idx, &value) in self.data[row].iter().enumerate() {
                if !value.is_zero() {
                    new_data.push(value);
                    new_indices.push(self.indices[row][idx]);
                }
            }

            self.data[row] = new_data;
            self.indices[row] = new_indices;
        }
    }

    fn sort_indices(&mut self) {
        // Call the implementation from the struct
        LilArray::sort_indices(self);
    }

    fn sorted_indices(&self) -> Box<dyn SparseArray<T>> {
        let mut sorted = self.clone();
        sorted.sort_indices();
        Box::new(sorted)
    }

    fn has_sorted_indices(&self) -> bool {
        // Check if each row has sorted indices
        for row in 0..self.shape.0 {
            if self.indices[row].len() > 1 {
                for i in 1..self.indices[row].len() {
                    if self.indices[row][i - 1] >= self.indices[row][i] {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn sum(&self, axis: Option<usize>) -> SparseResult<SparseSum<T>> {
        match axis {
            None => {
                // Sum over all elements
                let mut sum = T::zero();
                for row in 0..self.shape.0 {
                    for &val in self.data[row].iter() {
                        sum = sum + val;
                    }
                }
                Ok(SparseSum::Scalar(sum))
            }
            Some(0) => {
                // Sum over rows
                let (_, cols) = self.shape;
                let mut result = Array1::<T>::zeros(cols);

                for row in 0..self.shape.0 {
                    for (idx, &col) in self.indices[row].iter().enumerate() {
                        result[col] = result[col] + self.data[row][idx];
                    }
                }

                // Create a 1 x cols sparse array
                let mut lil = LilArray::new((1, cols));
                for (col, &val) in result.iter().enumerate() {
                    if !val.is_zero() {
                        lil.set(0, col, val)?;
                    }
                }

                Ok(SparseSum::SparseArray(Box::new(lil)))
            }
            Some(1) => {
                // Sum over columns
                let (rows, _) = self.shape;
                let mut result = Array1::<T>::zeros(rows);

                for row in 0..rows {
                    for &val in self.data[row].iter() {
                        result[row] = result[row] + val;
                    }
                }

                // Create a rows x 1 sparse array
                let mut lil = LilArray::new((rows, 1));
                for (row, &val) in result.iter().enumerate() {
                    if !val.is_zero() {
                        lil.set(row, 0, val)?;
                    }
                }

                Ok(SparseSum::SparseArray(Box::new(lil)))
            }
            _ => Err(SparseError::InvalidAxis),
        }
    }

    fn max(&self) -> T {
        if self.nnz() == 0 {
            return T::neg_infinity();
        }

        let mut max_val = T::neg_infinity();
        for row in 0..self.shape.0 {
            for &val in self.data[row].iter() {
                if val > max_val {
                    max_val = val;
                }
            }
        }

        // If matrix is not entirely filled and max is negative, zero is the max
        if max_val < T::zero() && self.nnz() < self.shape.0 * self.shape.1 {
            T::zero()
        } else {
            max_val
        }
    }

    fn min(&self) -> T {
        if self.nnz() == 0 {
            return T::infinity();
        }

        let mut min_val = T::infinity();
        for row in 0..self.shape.0 {
            for &val in self.data[row].iter() {
                if val < min_val {
                    min_val = val;
                }
            }
        }

        // If matrix is not entirely filled and min is positive, zero is the min
        if min_val > T::zero() && self.nnz() < self.shape.0 * self.shape.1 {
            T::zero()
        } else {
            min_val
        }
    }

    fn find(&self) -> (Array1<usize>, Array1<usize>, Array1<T>) {
        let nnz = self.nnz();
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for row in 0..self.shape.0 {
            for (idx, &col) in self.indices[row].iter().enumerate() {
                rows.push(row);
                cols.push(col);
                values.push(self.data[row][idx]);
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
            || start_row >= end_row
            || start_col >= end_col
        {
            return Err(SparseError::InvalidSliceRange);
        }

        let mut new_data = vec![Vec::new(); end_row - start_row];
        let mut new_indices = vec![Vec::new(); end_row - start_row];

        for row in start_row..end_row {
            for (idx, &col) in self.indices[row].iter().enumerate() {
                if col >= start_col && col < end_col {
                    new_data[row - start_row].push(self.data[row][idx]);
                    new_indices[row - start_row].push(col - start_col);
                }
            }
        }

        Ok(Box::new(LilArray {
            data: new_data,
            indices: new_indices,
            shape: (end_row - start_row, end_col - start_col),
        }))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<T> fmt::Debug for LilArray<T>
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
            "LilArray<{}x{}, nnz={}>",
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
    fn test_lil_array_construction() {
        // Create an empty LIL array
        let shape = (3, 3);
        let lil = LilArray::<f64>::new(shape);

        assert_eq!(lil.shape(), (3, 3));
        assert_eq!(lil.nnz(), 0);

        // Create from lists
        let data = vec![vec![1.0, 2.0], vec![3.0], vec![4.0, 5.0]];
        let indices = vec![vec![0, 2], vec![1], vec![0, 1]];

        let lil = LilArray::from_lists(data, indices, shape).unwrap();

        assert_eq!(lil.shape(), (3, 3));
        assert_eq!(lil.nnz(), 5);
        assert_eq!(lil.get(0, 0), 1.0);
        assert_eq!(lil.get(0, 2), 2.0);
        assert_eq!(lil.get(1, 1), 3.0);
        assert_eq!(lil.get(2, 0), 4.0);
        assert_eq!(lil.get(2, 1), 5.0);
        assert_eq!(lil.get(0, 1), 0.0);
    }

    #[test]
    fn test_lil_from_triplets() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let lil = LilArray::from_triplets(&rows, &cols, &data, shape).unwrap();

        assert_eq!(lil.shape(), (3, 3));
        assert_eq!(lil.nnz(), 5);
        assert_eq!(lil.get(0, 0), 1.0);
        assert_eq!(lil.get(0, 2), 2.0);
        assert_eq!(lil.get(1, 1), 3.0);
        assert_eq!(lil.get(2, 0), 4.0);
        assert_eq!(lil.get(2, 1), 5.0);
        assert_eq!(lil.get(0, 1), 0.0);
    }

    #[test]
    fn test_lil_array_to_array() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let lil = LilArray::from_triplets(&rows, &cols, &data, shape).unwrap();
        let dense = lil.to_array();

        assert_eq!(dense.shape(), &[3, 3]);
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[0, 1]], 0.0);
        assert_eq!(dense[[0, 2]], 2.0);
        assert_eq!(dense[[1, 0]], 0.0);
        assert_eq!(dense[[1, 1]], 3.0);
        assert_eq!(dense[[1, 2]], 0.0);
        assert_eq!(dense[[2, 0]], 4.0);
        assert_eq!(dense[[2, 1]], 5.0);
        assert_eq!(dense[[2, 2]], 0.0);
    }

    #[test]
    fn test_lil_set_get() {
        let mut lil = LilArray::<f64>::new((3, 3));

        // Set some values
        lil.set(0, 0, 1.0).unwrap();
        lil.set(0, 2, 2.0).unwrap();
        lil.set(1, 1, 3.0).unwrap();
        lil.set(2, 0, 4.0).unwrap();
        lil.set(2, 1, 5.0).unwrap();

        // Check values
        assert_eq!(lil.get(0, 0), 1.0);
        assert_eq!(lil.get(0, 2), 2.0);
        assert_eq!(lil.get(1, 1), 3.0);
        assert_eq!(lil.get(2, 0), 4.0);
        assert_eq!(lil.get(2, 1), 5.0);
        assert_eq!(lil.get(0, 1), 0.0);

        // Update a value
        lil.set(0, 0, 6.0).unwrap();
        assert_eq!(lil.get(0, 0), 6.0);

        // Set to zero (should remove the entry)
        lil.set(0, 0, 0.0).unwrap();
        assert_eq!(lil.get(0, 0), 0.0);
        assert_eq!(lil.nnz(), 4);
    }

    #[test]
    fn test_lil_to_csr() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let lil = LilArray::from_triplets(&rows, &cols, &data, shape).unwrap();

        // Convert to CSR
        let csr = lil.to_csr().unwrap();

        // Check values
        let dense = csr.to_array();
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[0, 1]], 0.0);
        assert_eq!(dense[[0, 2]], 2.0);
        assert_eq!(dense[[1, 0]], 0.0);
        assert_eq!(dense[[1, 1]], 3.0);
        assert_eq!(dense[[1, 2]], 0.0);
        assert_eq!(dense[[2, 0]], 4.0);
        assert_eq!(dense[[2, 1]], 5.0);
        assert_eq!(dense[[2, 2]], 0.0);
    }

    #[test]
    fn test_lil_to_coo() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let lil = LilArray::from_triplets(&rows, &cols, &data, shape).unwrap();

        // Convert to COO
        let coo = lil.to_coo().unwrap();

        // Check values
        let dense = coo.to_array();
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[0, 1]], 0.0);
        assert_eq!(dense[[0, 2]], 2.0);
        assert_eq!(dense[[1, 0]], 0.0);
        assert_eq!(dense[[1, 1]], 3.0);
        assert_eq!(dense[[1, 2]], 0.0);
        assert_eq!(dense[[2, 0]], 4.0);
        assert_eq!(dense[[2, 1]], 5.0);
        assert_eq!(dense[[2, 2]], 0.0);
    }

    #[test]
    fn test_lil_dot_vector() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let lil = LilArray::from_triplets(&rows, &cols, &data, shape).unwrap();

        // Create a vector
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Compute dot product
        let result = lil.dot_vector(&vector.view()).unwrap();

        // Check result: row[0]: 1.0*1.0 + 0.0*2.0 + 2.0*3.0 = 7.0
        //               row[1]: 0.0*1.0 + 3.0*2.0 + 0.0*3.0 = 6.0
        //               row[2]: 4.0*1.0 + 5.0*2.0 + 0.0*3.0 = 14.0
        assert_eq!(result.len(), 3);
        assert_relative_eq!(result[0], 7.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 6.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 14.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lil_eliminate_zeros() {
        let mut lil = LilArray::<f64>::new((2, 2));

        lil.set(0, 0, 1.0).unwrap();
        lil.set(0, 1, 0.0).unwrap(); // This won't actually add an entry
        lil.set(1, 0, 2.0).unwrap();
        lil.set(1, 1, 3.0).unwrap();

        // Set a value then zero it (this can leave an explicit zero in some implementations)
        lil.set(0, 1, 4.0).unwrap();
        lil.set(0, 1, 0.0).unwrap();

        // Manually insert a zero (normally this shouldn't happen)
        lil.data[1][0] = 0.0;

        assert_eq!(lil.nnz(), 3); // 3 entries, but one is zero

        // Eliminate zeros
        lil.eliminate_zeros();

        assert_eq!(lil.nnz(), 2); // Now only 2 entries (the non-zeros)
        assert_eq!(lil.get(0, 0), 1.0);
        assert_eq!(lil.get(1, 1), 3.0);
    }

    #[test]
    fn test_lil_sort_indices() {
        // Create a LIL array with unsorted indices
        let mut lil = LilArray::<f64>::new((2, 4));

        // Insert in non-sorted order
        lil.set(0, 3, 1.0).unwrap();
        lil.set(0, 1, 2.0).unwrap();
        lil.set(1, 2, 3.0).unwrap();
        lil.set(1, 0, 4.0).unwrap();

        // Manually mess up the sorting by directly swapping elements in the arrays
        // This is needed because LilArray.set() keeps indices sorted
        if lil.data[0].len() >= 2 {
            lil.data[0].swap(0, 1);
            lil.indices[0].swap(0, 1);
        }

        // Indices should be unsorted
        assert!(!lil.has_sorted_indices());

        // Sort indices
        lil.sort_indices();

        // Now indices should be sorted
        assert!(lil.has_sorted_indices());

        // Check values
        assert_eq!(lil.get(0, 1), 2.0);
        assert_eq!(lil.get(0, 3), 1.0);
        assert_eq!(lil.get(1, 0), 4.0);
        assert_eq!(lil.get(1, 2), 3.0);

        // Check internal structure
        assert_eq!(lil.indices[0][0], 1);
        assert_eq!(lil.indices[0][1], 3);
        assert_eq!(lil.data[0][0], 2.0);
        assert_eq!(lil.data[0][1], 1.0);

        assert_eq!(lil.indices[1][0], 0);
        assert_eq!(lil.indices[1][1], 2);
        assert_eq!(lil.data[1][0], 4.0);
        assert_eq!(lil.data[1][1], 3.0);
    }

    #[test]
    fn test_lil_slice() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let lil = LilArray::from_triplets(&rows, &cols, &data, shape).unwrap();

        // Get a slice
        let slice = lil.slice((1, 3), (0, 2)).unwrap();

        // Check slice shape
        assert_eq!(slice.shape(), (2, 2));

        // Check values
        assert_eq!(slice.get(0, 1), 3.0);
        assert_eq!(slice.get(1, 0), 4.0);
        assert_eq!(slice.get(1, 1), 5.0);
        assert_eq!(slice.get(0, 0), 0.0);
    }
}
