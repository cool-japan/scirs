// Sparse Array API
//
// This module provides the base trait for sparse arrays, inspired by SciPy's transition
// from matrix-based API to array-based API.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use crate::error::{SparseError, SparseResult};

/// Trait for sparse array types.
///
/// This trait defines the common interface for all sparse array implementations.
/// It is designed to align with SciPy's sparse array API, providing array-like semantics
/// rather than matrix-like semantics.
///
/// # Notes
///
/// The sparse array API differs from the sparse matrix API in the following ways:
///
/// - `*` operator performs element-wise multiplication, not matrix multiplication
/// - Matrix multiplication is done with the `dot` method or `@` operator in Python
/// - Operations like `sum` produce arrays, not matrices
/// - Sparse arrays use array-style slicing operations
///
pub trait SparseArray<T>: std::any::Any
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
    /// Returns the shape of the sparse array.
    fn shape(&self) -> (usize, usize);

    /// Returns the number of stored (non-zero) elements.
    fn nnz(&self) -> usize;

    /// Returns the data type of the sparse array.
    fn dtype(&self) -> &str;

    /// Returns a view of the sparse array as a dense ndarray.
    fn to_array(&self) -> Array2<T>;

    /// Returns a dense copy of the sparse array.
    fn toarray(&self) -> Array2<T>;

    /// Returns a sparse array in COO format.
    fn to_coo(&self) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Returns a sparse array in CSR format.
    fn to_csr(&self) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Returns a sparse array in CSC format.
    fn to_csc(&self) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Returns a sparse array in DOK format.
    fn to_dok(&self) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Returns a sparse array in LIL format.
    fn to_lil(&self) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Returns a sparse array in DIA format.
    fn to_dia(&self) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Returns a sparse array in BSR format.
    fn to_bsr(&self) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Element-wise addition.
    fn add(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Element-wise subtraction.
    fn sub(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Element-wise multiplication.
    fn mul(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Element-wise division.
    fn div(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Matrix multiplication.
    fn dot(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Matrix-vector multiplication.
    fn dot_vector(&self, other: &ArrayView1<T>) -> SparseResult<Array1<T>>;

    /// Transpose the sparse array.
    fn transpose(&self) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Return a copy of the sparse array with the specified elements.
    fn copy(&self) -> Box<dyn SparseArray<T>>;

    /// Get a value at the specified position.
    fn get(&self, i: usize, j: usize) -> T;

    /// Set a value at the specified position.
    fn set(&mut self, i: usize, j: usize, value: T) -> SparseResult<()>;

    /// Eliminate zeros from the sparse array.
    fn eliminate_zeros(&mut self);

    /// Sort indices of the sparse array.
    fn sort_indices(&mut self);

    /// Return a sorted copy of this sparse array.
    fn sorted_indices(&self) -> Box<dyn SparseArray<T>>;

    /// Check if indices are sorted.
    fn has_sorted_indices(&self) -> bool;

    /// Sum the sparse array elements.
    ///
    /// Parameters:
    /// - `axis`: The axis along which to sum. If None, sum over both axes.
    ///
    /// Returns a sparse array if summing over a single axis, or a scalar if summing over both axes.
    fn sum(&self, axis: Option<usize>) -> SparseResult<SparseSum<T>>;

    /// Compute the maximum value of the sparse array elements.
    fn max(&self) -> T;

    /// Compute the minimum value of the sparse array elements.
    fn min(&self) -> T;

    /// Return the indices and values of the nonzero elements.
    fn find(&self) -> (Array1<usize>, Array1<usize>, Array1<T>);

    /// Return a slice of the sparse array.
    fn slice(
        &self,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> SparseResult<Box<dyn SparseArray<T>>>;

    /// Returns the concrete type of the array for downcasting.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Returns the indptr array for CSR/CSC formats.
    /// For formats that don't have indptr, returns None.
    fn get_indptr(&self) -> Option<&Array1<usize>> {
        None
    }

    /// Returns the indptr array for CSR/CSC formats.
    /// For formats that don't have indptr, returns None.
    fn indptr(&self) -> Option<&Array1<usize>> {
        None
    }
}

/// Represents the result of a sum operation on a sparse array.
// Manually implement Debug and Clone instead of deriving them
pub enum SparseSum<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Sum over a single axis, returning a sparse array.
    SparseArray(Box<dyn SparseArray<T>>),

    /// Sum over both axes, returning a scalar.
    Scalar(T),
}

impl<T> Debug for SparseSum<T>
where
    T: Float + Debug + Copy + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SparseSum::SparseArray(_) => write!(f, "SparseSum::SparseArray(...)"),
            SparseSum::Scalar(value) => write!(f, "SparseSum::Scalar({value:?})"),
        }
    }
}

impl<T> Clone for SparseSum<T>
where
    T: Float + Debug + Copy + 'static,
{
    fn clone(&self) -> Self {
        match self {
            SparseSum::SparseArray(array) => SparseSum::SparseArray(array.copy()),
            SparseSum::Scalar(value) => SparseSum::Scalar(*value),
        }
    }
}

/// Identifies sparse arrays (both matrix and array types)
#[allow(dead_code)]
pub fn is_sparse<T>(obj: &dyn SparseArray<T>) -> bool
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
    true // Since this is a trait method, any object that implements it is sparse
}

/// Create a base SparseArray implementation for demonstrations and testing
pub struct SparseArrayBase<T>
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
    data: Array2<T>,
}

impl<T> SparseArrayBase<T>
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
    /// Create a new SparseArrayBase from a dense ndarray.
    pub fn new(data: Array2<T>) -> Self {
        Self { data }
    }
}

impl<T> SparseArray<T> for SparseArrayBase<T>
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
        let shape = self.data.shape();
        (shape[0], shape[1])
    }

    fn nnz(&self) -> usize {
        self.data.iter().filter(|&&x| !x.is_zero()).count()
    }

    fn dtype(&self) -> &str {
        "float" // This is a placeholder; ideally, we'd return the actual type
    }

    fn to_array(&self) -> Array2<T> {
        self.data.clone()
    }

    fn toarray(&self) -> Array2<T> {
        self.data.clone()
    }

    fn to_coo(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // In a real implementation, this would convert to COO format
        Ok(Box::new(self.clone()))
    }

    fn to_csr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // In a real implementation, this would convert to CSR format
        Ok(Box::new(self.clone()))
    }

    fn to_csc(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // In a real implementation, this would convert to CSC format
        Ok(Box::new(self.clone()))
    }

    fn to_dok(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // In a real implementation, this would convert to DOK format
        Ok(Box::new(self.clone()))
    }

    fn to_lil(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // In a real implementation, this would convert to LIL format
        Ok(Box::new(self.clone()))
    }

    fn to_dia(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // In a real implementation, this would convert to DIA format
        Ok(Box::new(self.clone()))
    }

    fn to_bsr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // In a real implementation, this would convert to BSR format
        Ok(Box::new(self.clone()))
    }

    fn add(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        let other_array = other.to_array();
        let result = &self.data + &other_array;
        Ok(Box::new(SparseArrayBase::new(result)))
    }

    fn sub(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        let other_array = other.to_array();
        let result = &self.data - &other_array;
        Ok(Box::new(SparseArrayBase::new(result)))
    }

    fn mul(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        let other_array = other.to_array();
        let result = &self.data * &other_array;
        Ok(Box::new(SparseArrayBase::new(result)))
    }

    fn div(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        let other_array = other.to_array();
        let result = &self.data / &other_array;
        Ok(Box::new(SparseArrayBase::new(result)))
    }

    fn dot(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        let other_array = other.to_array();
        let (m, n) = self.shape();
        let (p, q) = other.shape();

        if n != p {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: p,
            });
        }

        let mut result = Array2::zeros((m, q));
        for i in 0..m {
            for j in 0..q {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum + self.data[[i, k]] * other_array[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }

        Ok(Box::new(SparseArrayBase::new(result)))
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
        for i in 0..m {
            let mut sum = T::zero();
            for j in 0..n {
                sum = sum + self.data[[i, j]] * other[j];
            }
            result[i] = sum;
        }

        Ok(result)
    }

    fn transpose(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        Ok(Box::new(SparseArrayBase::new(self.data.t().to_owned())))
    }

    fn copy(&self) -> Box<dyn SparseArray<T>> {
        Box::new(self.clone())
    }

    fn get(&self, i: usize, j: usize) -> T {
        self.data[[i, j]]
    }

    fn set(&mut self, i: usize, j: usize, value: T) -> SparseResult<()> {
        let (m, n) = self.shape();
        if i >= m || j >= n {
            return Err(SparseError::IndexOutOfBounds {
                index: (i, j),
                shape: (m, n),
            });
        }
        self.data[[i, j]] = value;
        Ok(())
    }

    fn eliminate_zeros(&mut self) {
        // No-op for dense array
    }

    fn sort_indices(&mut self) {
        // No-op for dense array
    }

    fn sorted_indices(&self) -> Box<dyn SparseArray<T>> {
        self.copy()
    }

    fn has_sorted_indices(&self) -> bool {
        true // Dense array has implicitly sorted indices
    }

    fn sum(&self, axis: Option<usize>) -> SparseResult<SparseSum<T>> {
        match axis {
            None => {
                let mut sum = T::zero();
                for &val in self.data.iter() {
                    sum = sum + val;
                }
                Ok(SparseSum::Scalar(sum))
            }
            Some(0) => {
                let (_, n) = self.shape();
                let mut result = Array2::zeros((1, n));
                for j in 0..n {
                    let mut sum = T::zero();
                    for i in 0..self.data.shape()[0] {
                        sum = sum + self.data[[i, j]];
                    }
                    result[[0, j]] = sum;
                }
                Ok(SparseSum::SparseArray(Box::new(SparseArrayBase::new(
                    result,
                ))))
            }
            Some(1) => {
                let (m_, _) = self.shape();
                let mut result = Array2::zeros((m_, 1));
                for i in 0..m_ {
                    let mut sum = T::zero();
                    for j in 0..self.data.shape()[1] {
                        sum = sum + self.data[[i, j]];
                    }
                    result[[i, 0]] = sum;
                }
                Ok(SparseSum::SparseArray(Box::new(SparseArrayBase::new(
                    result,
                ))))
            }
            _ => Err(SparseError::InvalidAxis),
        }
    }

    fn max(&self) -> T {
        self.data
            .iter()
            .fold(T::neg_infinity(), |acc, &x| acc.max(x))
    }

    fn min(&self) -> T {
        self.data.iter().fold(T::infinity(), |acc, &x| acc.min(x))
    }

    fn find(&self) -> (Array1<usize>, Array1<usize>, Array1<T>) {
        let (m, n) = self.shape();
        let nnz = self.nnz();
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for i in 0..m {
            for j in 0..n {
                let value = self.data[[i, j]];
                if !value.is_zero() {
                    rows.push(i);
                    cols.push(j);
                    values.push(value);
                }
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
        let (m, n) = self.shape();

        if start_row >= m
            || end_row > m
            || start_col >= n
            || end_col > n
            || start_row >= end_row
            || start_col >= end_col
        {
            return Err(SparseError::InvalidSliceRange);
        }

        let view = self.data.slice(scirs2_core::ndarray::s![
            start_row..end_row,
            start_col..end_col
        ]);
        Ok(Box::new(SparseArrayBase::new(view.to_owned())))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<T> Clone for SparseArrayBase<T>
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
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_sparse_array_base() {
        let data = Array::from_shape_vec((3, 3), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0])
            .unwrap();
        let sparse = SparseArrayBase::new(data);

        assert_eq!(sparse.shape(), (3, 3));
        assert_eq!(sparse.nnz(), 5);
        assert_eq!(sparse.get(0, 0), 1.0);
        assert_eq!(sparse.get(1, 1), 3.0);
        assert_eq!(sparse.get(2, 2), 5.0);
        assert_eq!(sparse.get(0, 1), 0.0);
    }

    #[test]
    fn test_sparse_array_operations() {
        let data1 = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let data2 = Array::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let sparse1 = SparseArrayBase::new(data1);
        let sparse2 = SparseArrayBase::new(data2);

        // Test add
        let result = sparse1.add(&sparse2).unwrap();
        let result_array = result.to_array();
        assert_eq!(result_array[[0, 0]], 6.0);
        assert_eq!(result_array[[0, 1]], 8.0);
        assert_eq!(result_array[[1, 0]], 10.0);
        assert_eq!(result_array[[1, 1]], 12.0);

        // Test dot
        let result = sparse1.dot(&sparse2).unwrap();
        let result_array = result.to_array();
        assert_eq!(result_array[[0, 0]], 19.0);
        assert_eq!(result_array[[0, 1]], 22.0);
        assert_eq!(result_array[[1, 0]], 43.0);
        assert_eq!(result_array[[1, 1]], 50.0);
    }
}
