// Dictionary of Keys (DOK) Array implementation
//
// This module provides the DOK (Dictionary of Keys) array format,
// which is efficient for incremental construction of sparse arrays.

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use crate::coo_array::CooArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::{SparseArray, SparseSum};

/// DOK Array format
///
/// The DOK (Dictionary of Keys) format stores a sparse array in a dictionary (HashMap)
/// mapping (row, col) coordinate tuples to values.
///
/// # Notes
///
/// - Efficient for incremental construction (setting elements one by one)
/// - Fast random access to individual elements (get/set)
/// - Slow operations that require iterating over all elements
/// - Slow arithmetic operations
/// - Not suitable for large-scale computational operations
///
#[derive(Clone)]
pub struct DokArray<T>
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
    /// Dictionary mapping (row, col) to value
    data: HashMap<(usize, usize), T>,
    /// Shape of the sparse array
    shape: (usize, usize),
}

impl<T> DokArray<T>
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
    /// Creates a new DOK array with the given shape
    ///
    /// # Arguments
    /// * `shape` - Shape of the sparse array (rows, cols)
    ///
    /// # Returns
    /// A new empty `DokArray`
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            data: HashMap::new(),
            shape,
        }
    }

    /// Creates a DOK array from triplet format (COO-like)
    ///
    /// # Arguments
    /// * `rows` - Row indices
    /// * `cols` - Column indices
    /// * `data` - Values
    /// * `shape` - Shape of the sparse array
    ///
    /// # Returns
    /// A new `DokArray`
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

        let mut dok = Self::new(shape);
        for i in 0..rows.len() {
            if rows[i] >= shape.0 || cols[i] >= shape.1 {
                return Err(SparseError::IndexOutOfBounds {
                    index: (rows[i], cols[i]),
                    shape,
                });
            }
            // Only set non-zero values
            if !data[i].is_zero() {
                dok.data.insert((rows[i], cols[i]), data[i]);
            }
        }

        Ok(dok)
    }

    /// Returns a reference to the internal HashMap
    pub fn get_data(&self) -> &HashMap<(usize, usize), T> {
        &self.data
    }

    /// Returns the triplet representation (row indices, column indices, data)
    pub fn to_triplets(&self) -> (Array1<usize>, Array1<usize>, Array1<T>) {
        let nnz = self.nnz();
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        // Sort by row, then column for deterministic output
        let mut entries: Vec<_> = self.data.iter().collect();
        entries.sort_by_key(|&(&(row, col), _)| (row, col));

        for (&(row, col), &value) in entries {
            row_indices.push(row);
            col_indices.push(col);
            values.push(value);
        }

        (
            Array1::from_vec(row_indices),
            Array1::from_vec(col_indices),
            Array1::from_vec(values),
        )
    }

    /// Creates a DOK array from a dense ndarray
    ///
    /// # Arguments
    /// * `array` - Dense ndarray
    ///
    /// # Returns
    /// A new `DokArray` containing non-zero elements from the input array
    pub fn from_array(array: &Array2<T>) -> Self {
        let shape = (array.shape()[0], array.shape()[1]);
        let mut dok = Self::new(shape);

        for ((i, j), &value) in array.indexed_iter() {
            if !value.is_zero() {
                dok.data.insert((i, j), value);
            }
        }

        dok
    }
}

impl<T> SparseArray<T> for DokArray<T>
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
        "float" // This is a placeholder; ideally, we'd return the actual type
    }

    fn to_array(&self) -> Array2<T> {
        let (rows, cols) = self.shape;
        let mut result = Array2::zeros((rows, cols));

        for (&(row, col), &value) in &self.data {
            result[[row, col]] = value;
        }

        result
    }

    fn toarray(&self) -> Array2<T> {
        self.to_array()
    }

    fn to_coo(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (row_indices, col_indices, data) = self.to_triplets();
        CooArray::new(data, row_indices, col_indices, self.shape, true)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_csr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // First convert to COO, then to CSR
        match self.to_coo() {
            Ok(coo) => coo.to_csr(),
            Err(e) => Err(e),
        }
    }

    fn to_csc(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // First convert to COO, then to CSC
        match self.to_coo() {
            Ok(coo) => coo.to_csc(),
            Err(e) => Err(e),
        }
    }

    fn to_dok(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // We're already a DOK array
        Ok(Box::new(self.clone()))
    }

    fn to_lil(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        Err(SparseError::NotImplemented(
            "Conversion to LIL array".to_string(),
        ))
    }

    fn to_dia(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        Err(SparseError::NotImplemented(
            "Conversion to DIA array".to_string(),
        ))
    }

    fn to_bsr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        Err(SparseError::NotImplemented(
            "Conversion to BSR array".to_string(),
        ))
    }

    fn add(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        if self.shape() != other.shape() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape().0,
                found: other.shape().0,
            });
        }

        let mut result = self.clone();
        let other_array = other.to_array();

        // Add existing values from self
        for (&(row, col), &value) in &self.data {
            result.set(row, col, value + other_array[[row, col]])?;
        }

        // Add values from other that aren't in self
        for ((row, col), &value) in other_array.indexed_iter() {
            if !self.data.contains_key(&(row, col)) && !value.is_zero() {
                result.set(row, col, value)?;
            }
        }

        Ok(Box::new(result))
    }

    fn sub(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        if self.shape() != other.shape() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape().0,
                found: other.shape().0,
            });
        }

        let mut result = self.clone();
        let other_array = other.to_array();

        // Subtract existing values from self
        for (&(row, col), &value) in &self.data {
            result.set(row, col, value - other_array[[row, col]])?;
        }

        // Subtract values from other that aren't in self
        for ((row, col), &value) in other_array.indexed_iter() {
            if !self.data.contains_key(&(row, col)) && !value.is_zero() {
                result.set(row, col, -value)?;
            }
        }

        Ok(Box::new(result))
    }

    fn mul(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        if self.shape() != other.shape() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape().0,
                found: other.shape().0,
            });
        }

        let mut result = DokArray::new(self.shape());
        let other_array = other.to_array();

        // Only need to process entries in self
        // since a*0 = 0 for any a
        for (&(row, col), &value) in &self.data {
            let product = value * other_array[[row, col]];
            if !product.is_zero() {
                result.set(row, col, product)?;
            }
        }

        Ok(Box::new(result))
    }

    fn div(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        if self.shape() != other.shape() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape().0,
                found: other.shape().0,
            });
        }

        let mut result = DokArray::new(self.shape());
        let other_array = other.to_array();

        for (&(row, col), &value) in &self.data {
            let divisor = other_array[[row, col]];
            if divisor.is_zero() {
                return Err(SparseError::ComputationError(
                    "Division by zero".to_string(),
                ));
            }

            let quotient = value / divisor;
            if !quotient.is_zero() {
                result.set(row, col, quotient)?;
            }
        }

        Ok(Box::new(result))
    }

    fn dot(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (_m, n) = self.shape();
        let (p, _q) = other.shape();

        if n != p {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: p,
            });
        }

        // Convert to CSR for efficient matrix multiplication
        let csr_self = self.to_csr()?;
        let csr_other = other.to_csr()?;

        csr_self.dot(&*csr_other)
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

        for (&(row, col), &value) in &self.data {
            result[row] = result[row] + value * other[col];
        }

        Ok(result)
    }

    fn transpose(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (rows, cols) = self.shape;
        let mut result = DokArray::new((cols, rows));

        for (&(row, col), &value) in &self.data {
            result.set(col, row, value)?;
        }

        Ok(Box::new(result))
    }

    fn copy(&self) -> Box<dyn SparseArray<T>> {
        Box::new(self.clone())
    }

    fn get(&self, i: usize, j: usize) -> T {
        if i >= self.shape.0 || j >= self.shape.1 {
            return T::zero();
        }

        *self.data.get(&(i, j)).unwrap_or(&T::zero())
    }

    fn set(&mut self, i: usize, j: usize, value: T) -> SparseResult<()> {
        if i >= self.shape.0 || j >= self.shape.1 {
            return Err(SparseError::IndexOutOfBounds {
                index: (i, j),
                shape: self.shape,
            });
        }

        if value.is_zero() {
            // Remove zero entries
            self.data.remove(&(i, j));
        } else {
            // Set non-zero value
            self.data.insert((i, j), value);
        }

        Ok(())
    }

    fn eliminate_zeros(&mut self) {
        // DOK format already doesn't store zeros, but just in case
        self.data.retain(|_, &mut value| !value.is_zero());
    }

    fn sort_indices(&mut self) {
        // No-op for DOK format since it's a HashMap
    }

    fn sorted_indices(&self) -> Box<dyn SparseArray<T>> {
        // DOK doesn't have the concept of sorted indices
        self.copy()
    }

    fn has_sorted_indices(&self) -> bool {
        true // DOK format doesn't have the concept of sorted indices
    }

    fn sum(&self, axis: Option<usize>) -> SparseResult<SparseSum<T>> {
        match axis {
            None => {
                // Sum all elements
                let mut sum = T::zero();
                for &value in self.data.values() {
                    sum = sum + value;
                }
                Ok(SparseSum::Scalar(sum))
            }
            Some(0) => {
                // Sum along rows
                let (_, cols) = self.shape();
                let mut result = DokArray::new((1, cols));

                for (&(_row, col), &value) in &self.data {
                    let current = result.get(0, col);
                    result.set(0, col, current + value)?;
                }

                Ok(SparseSum::SparseArray(Box::new(result)))
            }
            Some(1) => {
                // Sum along columns
                let (rows, _) = self.shape();
                let mut result = DokArray::new((rows, 1));

                for (&(row, _col), &value) in &self.data {
                    let current = result.get(row, 0);
                    result.set(row, 0, current + value)?;
                }

                Ok(SparseSum::SparseArray(Box::new(result)))
            }
            _ => Err(SparseError::InvalidAxis),
        }
    }

    fn max(&self) -> T {
        if self.data.is_empty() {
            return T::nan();
        }

        self.data
            .values()
            .fold(T::neg_infinity(), |acc, &x| acc.max(x))
    }

    fn min(&self) -> T {
        if self.data.is_empty() {
            return T::nan();
        }

        self.data.values().fold(T::infinity(), |acc, &x| acc.min(x))
    }

    fn find(&self) -> (Array1<usize>, Array1<usize>, Array1<T>) {
        self.to_triplets()
    }

    fn slice(
        &self,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (start_row, end_row) = row_range;
        let (start_col, end_col) = col_range;
        let (rows, cols) = self.shape;

        if start_row >= rows
            || end_row > rows
            || start_col >= cols
            || end_col > cols
            || start_row >= end_row
            || start_col >= end_col
        {
            return Err(SparseError::InvalidSliceRange);
        }

        let slice_shape = (end_row - start_row, end_col - start_col);
        let mut result = DokArray::new(slice_shape);

        for (&(row, col), &value) in &self.data {
            if row >= start_row && row < end_row && col >= start_col && col < end_col {
                result.set(row - start_row, col - start_col, value)?;
            }
        }

        Ok(Box::new(result))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_dok_array_create_and_access() {
        // Create a 3x3 sparse array
        let mut array = DokArray::<f64>::new((3, 3));

        // Set some values
        array.set(0, 0, 1.0).unwrap();
        array.set(0, 2, 2.0).unwrap();
        array.set(1, 2, 3.0).unwrap();
        array.set(2, 0, 4.0).unwrap();
        array.set(2, 1, 5.0).unwrap();

        assert_eq!(array.nnz(), 5);

        // Access values
        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(0, 1), 0.0); // Zero entry
        assert_eq!(array.get(0, 2), 2.0);
        assert_eq!(array.get(1, 2), 3.0);
        assert_eq!(array.get(2, 0), 4.0);
        assert_eq!(array.get(2, 1), 5.0);

        // Set a value to zero should remove it
        array.set(0, 0, 0.0).unwrap();
        assert_eq!(array.nnz(), 4);
        assert_eq!(array.get(0, 0), 0.0);

        // Out of bounds access should return zero
        assert_eq!(array.get(3, 0), 0.0);
        assert_eq!(array.get(0, 3), 0.0);
    }

    #[test]
    fn test_dok_array_from_triplets() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let array = DokArray::from_triplets(&rows, &cols, &data, (3, 3)).unwrap();

        assert_eq!(array.nnz(), 5);
        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(0, 2), 2.0);
        assert_eq!(array.get(1, 2), 3.0);
        assert_eq!(array.get(2, 0), 4.0);
        assert_eq!(array.get(2, 1), 5.0);
    }

    #[test]
    fn test_dok_array_to_array() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let array = DokArray::from_triplets(&rows, &cols, &data, (3, 3)).unwrap();
        let dense = array.to_array();

        let expected =
            Array::from_shape_vec((3, 3), vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0])
                .unwrap();

        assert_eq!(dense, expected);
    }

    #[test]
    fn test_dok_array_from_array() {
        let dense =
            Array::from_shape_vec((3, 3), vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0])
                .unwrap();

        let array = DokArray::from_array(&dense);

        assert_eq!(array.nnz(), 5);
        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(0, 2), 2.0);
        assert_eq!(array.get(1, 2), 3.0);
        assert_eq!(array.get(2, 0), 4.0);
        assert_eq!(array.get(2, 1), 5.0);
    }

    #[test]
    fn test_dok_array_add() {
        let mut array1 = DokArray::<f64>::new((2, 2));
        array1.set(0, 0, 1.0).unwrap();
        array1.set(0, 1, 2.0).unwrap();
        array1.set(1, 0, 3.0).unwrap();

        let mut array2 = DokArray::<f64>::new((2, 2));
        array2.set(0, 0, 4.0).unwrap();
        array2.set(1, 1, 5.0).unwrap();

        let result = array1.add(&array2).unwrap();
        let dense_result = result.to_array();

        assert_eq!(dense_result[[0, 0]], 5.0);
        assert_eq!(dense_result[[0, 1]], 2.0);
        assert_eq!(dense_result[[1, 0]], 3.0);
        assert_eq!(dense_result[[1, 1]], 5.0);
    }

    #[test]
    fn test_dok_array_mul() {
        let mut array1 = DokArray::<f64>::new((2, 2));
        array1.set(0, 0, 1.0).unwrap();
        array1.set(0, 1, 2.0).unwrap();
        array1.set(1, 0, 3.0).unwrap();
        array1.set(1, 1, 4.0).unwrap();

        let mut array2 = DokArray::<f64>::new((2, 2));
        array2.set(0, 0, 5.0).unwrap();
        array2.set(0, 1, 6.0).unwrap();
        array2.set(1, 0, 7.0).unwrap();
        array2.set(1, 1, 8.0).unwrap();

        // Element-wise multiplication
        let result = array1.mul(&array2).unwrap();
        let dense_result = result.to_array();

        assert_eq!(dense_result[[0, 0]], 5.0);
        assert_eq!(dense_result[[0, 1]], 12.0);
        assert_eq!(dense_result[[1, 0]], 21.0);
        assert_eq!(dense_result[[1, 1]], 32.0);
    }

    #[test]
    fn test_dok_array_dot() {
        let mut array1 = DokArray::<f64>::new((2, 2));
        array1.set(0, 0, 1.0).unwrap();
        array1.set(0, 1, 2.0).unwrap();
        array1.set(1, 0, 3.0).unwrap();
        array1.set(1, 1, 4.0).unwrap();

        let mut array2 = DokArray::<f64>::new((2, 2));
        array2.set(0, 0, 5.0).unwrap();
        array2.set(0, 1, 6.0).unwrap();
        array2.set(1, 0, 7.0).unwrap();
        array2.set(1, 1, 8.0).unwrap();

        // Matrix multiplication
        let result = array1.dot(&array2).unwrap();
        let dense_result = result.to_array();

        // [1 2] [5 6] = [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3 4] [7 8]   [3*5 + 4*7, 3*6 + 4*8]   [43, 50]
        assert_eq!(dense_result[[0, 0]], 19.0);
        assert_eq!(dense_result[[0, 1]], 22.0);
        assert_eq!(dense_result[[1, 0]], 43.0);
        assert_eq!(dense_result[[1, 1]], 50.0);
    }

    #[test]
    fn test_dok_array_transpose() {
        let mut array = DokArray::<f64>::new((2, 3));
        array.set(0, 0, 1.0).unwrap();
        array.set(0, 1, 2.0).unwrap();
        array.set(0, 2, 3.0).unwrap();
        array.set(1, 0, 4.0).unwrap();
        array.set(1, 1, 5.0).unwrap();
        array.set(1, 2, 6.0).unwrap();

        let transposed = array.transpose().unwrap();

        assert_eq!(transposed.shape(), (3, 2));
        assert_eq!(transposed.get(0, 0), 1.0);
        assert_eq!(transposed.get(1, 0), 2.0);
        assert_eq!(transposed.get(2, 0), 3.0);
        assert_eq!(transposed.get(0, 1), 4.0);
        assert_eq!(transposed.get(1, 1), 5.0);
        assert_eq!(transposed.get(2, 1), 6.0);
    }

    #[test]
    fn test_dok_array_slice() {
        let mut array = DokArray::<f64>::new((3, 3));
        array.set(0, 0, 1.0).unwrap();
        array.set(0, 1, 2.0).unwrap();
        array.set(0, 2, 3.0).unwrap();
        array.set(1, 0, 4.0).unwrap();
        array.set(1, 1, 5.0).unwrap();
        array.set(1, 2, 6.0).unwrap();
        array.set(2, 0, 7.0).unwrap();
        array.set(2, 1, 8.0).unwrap();
        array.set(2, 2, 9.0).unwrap();

        let slice = array.slice((0, 2), (1, 3)).unwrap();

        assert_eq!(slice.shape(), (2, 2));
        assert_eq!(slice.get(0, 0), 2.0);
        assert_eq!(slice.get(0, 1), 3.0);
        assert_eq!(slice.get(1, 0), 5.0);
        assert_eq!(slice.get(1, 1), 6.0);
    }

    #[test]
    fn test_dok_array_sum() {
        let mut array = DokArray::<f64>::new((2, 3));
        array.set(0, 0, 1.0).unwrap();
        array.set(0, 1, 2.0).unwrap();
        array.set(0, 2, 3.0).unwrap();
        array.set(1, 0, 4.0).unwrap();
        array.set(1, 1, 5.0).unwrap();
        array.set(1, 2, 6.0).unwrap();

        // Sum all elements
        match array.sum(None).unwrap() {
            SparseSum::Scalar(sum) => assert_eq!(sum, 21.0),
            _ => panic!("Expected scalar sum"),
        }

        // Sum along rows (axis 0)
        match array.sum(Some(0)).unwrap() {
            SparseSum::SparseArray(sum_array) => {
                assert_eq!(sum_array.shape(), (1, 3));
                assert_eq!(sum_array.get(0, 0), 5.0);
                assert_eq!(sum_array.get(0, 1), 7.0);
                assert_eq!(sum_array.get(0, 2), 9.0);
            }
            _ => panic!("Expected sparse array"),
        }

        // Sum along columns (axis 1)
        match array.sum(Some(1)).unwrap() {
            SparseSum::SparseArray(sum_array) => {
                assert_eq!(sum_array.shape(), (2, 1));
                assert_eq!(sum_array.get(0, 0), 6.0);
                assert_eq!(sum_array.get(1, 0), 15.0);
            }
            _ => panic!("Expected sparse array"),
        }
    }
}
