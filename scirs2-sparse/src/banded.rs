//! Banded matrix format (legacy matrix API)

use crate::banded_array::BandedArray;
use crate::error::SparseResult;
use crate::sparray::SparseArray;
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::{Float, One, Zero};
use std::fmt::Debug;

/// Legacy banded matrix wrapper around BandedArray
pub type BandedMatrix<T> = BandedArray<T>;

impl<T> BandedMatrix<T>
where
    T: Float
        + Debug
        + std::fmt::Display
        + Copy
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::AddAssign,
{
    /// Matrix multiplication (for legacy API compatibility)
    pub fn matmul(&self, other: &BandedMatrix<T>) -> SparseResult<BandedMatrix<T>> {
        // Convert to dense for multiplication, then back to banded
        let a_dense = self.to_array();
        let b_dense = other.to_array();

        if a_dense.ncols() != b_dense.nrows() {
            return Err(crate::error::SparseError::DimensionMismatch {
                expected: a_dense.ncols(),
                found: b_dense.nrows(),
            });
        }

        let result_dense = a_dense.dot(&b_dense);

        // Estimate bandwidth of result
        let max_bandwidth = self.kl() + self.ku() + other.kl() + other.ku();

        // Extract banded structure from result
        Self::from_dense(&result_dense, max_bandwidth, max_bandwidth)
    }

    /// Create banded matrix from dense array
    pub fn from_dense(dense: &Array2<T>, kl: usize, ku: usize) -> SparseResult<Self> {
        let (rows, cols) = dense.dim();
        let mut result = Self::zeros((rows, cols), kl, ku);

        for i in 0..rows {
            for j in 0..cols {
                if result.is_in_band(i, j) {
                    let val = dense[[i, j]];
                    if !val.is_zero() {
                        result.set_unchecked(i, j, val);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Get a mutable reference to an element (legacy API)
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if !self.is_in_band(row, col) {
            return None;
        }

        let band_idx = self.ku() + col - row;
        if band_idx < self.kl() + self.ku() + 1 && row < self.shape().0 {
            Some(&mut self.data_mut()[[band_idx, row]])
        } else {
            None
        }
    }

    /// Get mutable reference to data (private helper)  
    #[allow(dead_code)]
    fn banded_data_mut(&mut self) -> &mut Array2<T> {
        BandedArray::data_mut(self)
    }

    /// Set element (legacy API)
    pub fn set(&mut self, row: usize, col: usize, value: T) -> SparseResult<()> {
        if !self.is_in_band(row, col) {
            if !value.is_zero() {
                return Err(crate::error::SparseError::ValueError(format!(
                    "Cannot set non-zero element at ({row}, {col}) outside band structure"
                )));
            }
            return Ok(());
        }

        self.set_unchecked(row, col, value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_banded_matrix_creation() {
        let diag = vec![1.0, 2.0, 3.0];
        let lower = vec![4.0, 5.0];
        let upper = vec![6.0, 7.0];

        let matrix = BandedMatrix::tridiagonal(&diag, &lower, &upper).unwrap();

        assert_eq!(matrix.shape(), (3, 3));
        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(1, 1), 2.0);
        assert_eq!(matrix.get(2, 2), 3.0);
        assert_eq!(matrix.get(1, 0), 4.0);
        assert_eq!(matrix.get(2, 1), 5.0);
        assert_eq!(matrix.get(0, 1), 6.0);
        assert_eq!(matrix.get(1, 2), 7.0);
    }

    #[test]
    fn test_banded_matrix_set() {
        let mut matrix = BandedMatrix::<f64>::zeros((3, 3), 1, 1);

        // Should succeed for in-band elements
        assert!(matrix.set(0, 0, 1.0).is_ok());
        assert!(matrix.set(0, 1, 2.0).is_ok());
        assert!(matrix.set(1, 0, 3.0).is_ok());

        // Should fail for out-of-band non-zero elements
        assert!(matrix.set(0, 2, 4.0).is_err());

        // Should succeed for out-of-band zero elements
        assert!(matrix.set(0, 2, 0.0).is_ok());

        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(0, 1), 2.0);
        assert_eq!(matrix.get(1, 0), 3.0);
    }

    #[test]
    fn test_banded_matrix_matmul() {
        let a = BandedMatrix::tridiagonal(&[2.0, 2.0, 2.0], &[1.0, 1.0], &[1.0, 1.0]).unwrap();

        let b = BandedMatrix::tridiagonal(&[1.0, 1.0, 1.0], &[0.5, 0.5], &[0.5, 0.5]).unwrap();

        let c = a.matmul(&b).unwrap();

        // Verify some elements of the result
        assert!(c.shape() == (3, 3));

        // Manual verification for (0,0): 2*1 + 1*0.5 = 2.5
        assert_relative_eq!(c.get(0, 0), 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_from_dense() {
        let dense =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 0.0, 6.0, 7.0])
                .unwrap();

        let banded = BandedMatrix::from_dense(&dense, 1, 1).unwrap();

        assert_eq!(banded.get(0, 0), 1.0);
        assert_eq!(banded.get(0, 1), 2.0);
        assert_eq!(banded.get(1, 0), 3.0);
        assert_eq!(banded.get(1, 1), 4.0);
        assert_eq!(banded.get(1, 2), 5.0);
        assert_eq!(banded.get(2, 1), 6.0);
        assert_eq!(banded.get(2, 2), 7.0);

        // Out-of-band elements should be zero
        assert_eq!(banded.get(0, 2), 0.0);
        assert_eq!(banded.get(2, 0), 0.0);
    }
}
