//! Hankel matrix implementation
//!
//! A Hankel matrix is a matrix where each ascending diagonal from left to right
//! is constant. This means it has the form:
//!
//! ```text
//! [a_0    a_1    a_2    ...  a_{n-1}]
//! [a_1    a_2    a_3    ...  a_n    ]
//! [a_2    a_3    a_4    ...  a_{n+1}]
//! [  ...    ...    ...  ...    ...  ]
//! [a_{m-1} a_m    a_{m+1} ... a_{m+n-2}]
//! ```
//!
//! The entire matrix is determined by its first column and last row.

use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use std::{fmt::Debug, iter::Sum};

use super::StructuredMatrix;
use crate::error::{LinalgError, LinalgResult};

/// Hankel matrix implementation
///
/// A Hankel matrix is represented by its first column and last row,
/// where the last element of the first column must be the same as the
/// first element of the last row.
#[derive(Debug, Clone)]
pub struct HankelMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// First column of the Hankel matrix
    first_col: Array1<A>,
    /// Last row of the Hankel matrix
    last_row: Array1<A>,
    /// Number of rows
    nrows: usize,
    /// Number of columns
    ncols: usize,
}

impl<A> HankelMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// Create a new Hankel matrix from its first column and last row
    ///
    /// The last element of the first column must be the same as the first element of the last row.
    ///
    /// # Arguments
    ///
    /// * `first_col` - First column of the Hankel matrix
    /// * `last_row` - Last row of the Hankel matrix
    ///
    /// # Returns
    ///
    /// A new `HankelMatrix` instance
    pub fn new(_first_col: ArrayView1<A>, lastrow: ArrayView1<A>) -> LinalgResult<Self> {
        // Check that the first elements match
        if _first_col.is_empty() || lastrow.is_empty() {
            return Err(LinalgError::InvalidInputError(
                "Column and row must not be empty".to_string(),
            ));
        }

        if (_first_col[_first_col.len() - 1] - lastrow[0]).abs() > A::epsilon() {
            return Err(LinalgError::InvalidInputError(
                "Last element of first column must be the same as first element of last row"
                    .to_string(),
            ));
        }

        Ok(HankelMatrix {
            first_col: _first_col.to_owned(),
            last_row: lastrow.to_owned(),
            nrows: _first_col.len(),
            ncols: lastrow.len(),
        })
    }

    /// Create a new Hankel matrix from a single sequence
    ///
    /// The sequence will be used to form both the first column and last row.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The sequence forming the Hankel matrix
    /// * `n_rows` - Number of rows in the resulting matrix
    /// * `n_cols` - Number of columns in the resulting matrix
    ///
    /// # Returns
    ///
    /// A new `HankelMatrix` instance
    pub fn from_sequence(
        sequence: ArrayView1<A>,
        n_rows: usize,
        n_cols: usize,
    ) -> LinalgResult<Self> {
        if sequence.len() < n_rows + n_cols - 1 {
            return Err(LinalgError::InvalidInputError(format!(
                "Sequence length must be at least nrows + ncols - 1 = {}, got {}",
                n_rows + n_cols - 1,
                sequence.len()
            )));
        }

        let first_col = sequence
            .slice(scirs2_core::ndarray::s![0..n_rows])
            .to_owned();
        let last_row = sequence
            .slice(scirs2_core::ndarray::s![
                (n_rows - 1)..(n_rows + n_cols - 1)
            ])
            .to_owned();

        Ok(HankelMatrix {
            first_col,
            last_row,
            nrows: n_rows,
            ncols: n_cols,
        })
    }
}

impl<A> StructuredMatrix<A> for HankelMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn get(&self, i: usize, j: usize) -> LinalgResult<A> {
        if i >= self.nrows || j >= self.ncols {
            return Err(LinalgError::IndexError(format!(
                "Index out of bounds: ({}, {}) for matrix of shape {}x{}",
                i, j, self.nrows, self.ncols
            )));
        }

        let sum_idx = i + j;

        if sum_idx < self.nrows {
            // Use first_col if index is within bounds
            Ok(self.first_col[sum_idx])
        } else {
            // Otherwise use last_row
            let j_idx = sum_idx - self.nrows + 1;
            if j_idx < self.ncols {
                Ok(self.last_row[j_idx])
            } else {
                // Should never happen due to bounds check
                Err(LinalgError::IndexError(format!(
                    "Index out of bounds: sum index {sum_idx} exceeds matrix dimensions"
                )))
            }
        }
    }

    fn matvec(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if x.len() != self.ncols {
            return Err(LinalgError::ShapeError(format!(
                "Input vector has wrong length: expected {}, got {}",
                self.ncols,
                x.len()
            )));
        }

        let mut result = Array1::zeros(self.nrows);

        for i in 0..self.nrows {
            for j in 0..self.ncols {
                result[i] += self.get(i, j).unwrap() * x[j];
            }
        }

        Ok(result)
    }

    fn matvec_transpose(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if x.len() != self.nrows {
            return Err(LinalgError::ShapeError(format!(
                "Input vector has wrong length: expected {}, got {}",
                self.nrows,
                x.len()
            )));
        }

        let mut result = Array1::zeros(self.ncols);

        for j in 0..self.ncols {
            for i in 0..self.nrows {
                result[j] += self.get(i, j).unwrap() * x[i];
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_hankel_creation() {
        let first_col = array![1.0, 2.0, 3.0];
        let last_row = array![3.0, 4.0, 5.0];

        let hankel = HankelMatrix::new(first_col.view(), last_row.view()).unwrap();

        assert_eq!(hankel.nrows(), 3);
        assert_eq!(hankel.ncols(), 3);

        // Check the elements
        // Expected matrix:
        // [1 2 3]
        // [2 3 4]
        // [3 4 5]

        assert_relative_eq!(hankel.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(hankel.get(0, 1).unwrap(), 2.0);
        assert_relative_eq!(hankel.get(0, 2).unwrap(), 3.0);
        assert_relative_eq!(hankel.get(1, 0).unwrap(), 2.0);
        assert_relative_eq!(hankel.get(1, 1).unwrap(), 3.0);
        assert_relative_eq!(hankel.get(1, 2).unwrap(), 4.0);
        assert_relative_eq!(hankel.get(2, 0).unwrap(), 3.0);
        assert_relative_eq!(hankel.get(2, 1).unwrap(), 4.0);
        assert_relative_eq!(hankel.get(2, 2).unwrap(), 5.0);
    }

    #[test]
    fn test_hankel_from_sequence() {
        let sequence = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let hankel = HankelMatrix::from_sequence(sequence.view(), 3, 3).unwrap();

        assert_eq!(hankel.nrows(), 3);
        assert_eq!(hankel.ncols(), 3);

        // Check the elements
        // Expected matrix:
        // [1 2 3]
        // [2 3 4]
        // [3 4 5]

        assert_relative_eq!(hankel.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(hankel.get(0, 1).unwrap(), 2.0);
        assert_relative_eq!(hankel.get(0, 2).unwrap(), 3.0);
        assert_relative_eq!(hankel.get(1, 0).unwrap(), 2.0);
        assert_relative_eq!(hankel.get(1, 1).unwrap(), 3.0);
        assert_relative_eq!(hankel.get(1, 2).unwrap(), 4.0);
        assert_relative_eq!(hankel.get(2, 0).unwrap(), 3.0);
        assert_relative_eq!(hankel.get(2, 1).unwrap(), 4.0);
        assert_relative_eq!(hankel.get(2, 2).unwrap(), 5.0);
    }

    #[test]
    fn test_hankel_rectangular() {
        // Test rectangular Hankel matrix
        let first_col = array![1.0, 2.0, 3.0, 4.0];
        let last_row = array![4.0, 5.0, 6.0];

        let hankel = HankelMatrix::new(first_col.view(), last_row.view()).unwrap();

        assert_eq!(hankel.nrows(), 4);
        assert_eq!(hankel.ncols(), 3);

        // Check the elements
        // Expected matrix:
        // [1 2 3]
        // [2 3 4]
        // [3 4 5]
        // [4 5 6]

        assert_relative_eq!(hankel.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(hankel.get(0, 1).unwrap(), 2.0);
        assert_relative_eq!(hankel.get(0, 2).unwrap(), 3.0);

        assert_relative_eq!(hankel.get(1, 0).unwrap(), 2.0);
        assert_relative_eq!(hankel.get(1, 1).unwrap(), 3.0);
        assert_relative_eq!(hankel.get(1, 2).unwrap(), 4.0);

        assert_relative_eq!(hankel.get(2, 0).unwrap(), 3.0);
        assert_relative_eq!(hankel.get(2, 1).unwrap(), 4.0);
        assert_relative_eq!(hankel.get(2, 2).unwrap(), 5.0);

        assert_relative_eq!(hankel.get(3, 0).unwrap(), 4.0);
        assert_relative_eq!(hankel.get(3, 1).unwrap(), 5.0);
        assert_relative_eq!(hankel.get(3, 2).unwrap(), 6.0);
    }

    #[test]
    fn test_hankel_matvec() {
        let first_col = array![1.0, 2.0, 3.0];
        let last_row = array![3.0, 4.0, 5.0];

        let hankel = HankelMatrix::new(first_col.view(), last_row.view()).unwrap();

        // Expected matrix:
        // [1 2 3]
        // [2 3 4]
        // [3 4 5]

        let x = array![1.0, 2.0, 3.0];
        let y = hankel.matvec(&x.view()).unwrap();

        // Expected result: [1*1 + 2*2 + 3*3, 2*1 + 3*2 + 4*3, 3*1 + 4*2 + 5*3]
        //                = [14, 20, 26]
        assert_eq!(y.len(), 3);
        assert_relative_eq!(y[0], 14.0);
        assert_relative_eq!(y[1], 20.0);
        assert_relative_eq!(y[2], 26.0);
    }

    #[test]
    fn test_hankel_matvec_transpose() {
        let first_col = array![1.0, 2.0, 3.0];
        let last_row = array![3.0, 4.0, 5.0];

        let hankel = HankelMatrix::new(first_col.view(), last_row.view()).unwrap();

        // Transpose of the matrix:
        // [1 2 3]
        // [2 3 4]
        // [3 4 5]

        let x = array![1.0, 2.0, 3.0];
        let y = hankel.matvec_transpose(&x.view()).unwrap();

        // Expected result: [1*1 + 2*2 + 3*3, 2*1 + 3*2 + 4*3, 3*1 + 4*2 + 5*3]
        //                = [14, 20, 26]
        assert_eq!(y.len(), 3);
        assert_relative_eq!(y[0], 14.0);
        assert_relative_eq!(y[1], 20.0);
        assert_relative_eq!(y[2], 26.0);
    }

    #[test]
    fn test_hankel_to_dense() {
        let first_col = array![1.0, 2.0, 3.0];
        let last_row = array![3.0, 4.0, 5.0];

        let hankel = HankelMatrix::new(first_col.view(), last_row.view()).unwrap();

        let dense = hankel.to_dense().unwrap();

        assert_eq!(dense.shape(), &[3, 3]);

        // Expected matrix:
        // [1 2 3]
        // [2 3 4]
        // [3 4 5]

        assert_relative_eq!(dense[[0, 0]], 1.0);
        assert_relative_eq!(dense[[0, 1]], 2.0);
        assert_relative_eq!(dense[[0, 2]], 3.0);
        assert_relative_eq!(dense[[1, 0]], 2.0);
        assert_relative_eq!(dense[[1, 1]], 3.0);
        assert_relative_eq!(dense[[1, 2]], 4.0);
        assert_relative_eq!(dense[[2, 0]], 3.0);
        assert_relative_eq!(dense[[2, 1]], 4.0);
        assert_relative_eq!(dense[[2, 2]], 5.0);
    }

    #[test]
    fn test_invalid_inputs() {
        // Last element of first_col doesn't match first element of last_row
        let first_col = array![1.0, 2.0, 4.0]; // Last element is 4.0
        let last_row = array![3.0, 4.0, 5.0]; // First element is 3.0

        let result = HankelMatrix::<f64>::new(first_col.view(), last_row.view());
        assert!(result.is_err());

        // Empty arrays
        let first_col = array![];
        let last_row = array![];

        let result = HankelMatrix::<f64>::new(first_col.view(), last_row.view());
        assert!(result.is_err());

        // Sequence too short for from_sequence
        let sequence = array![1.0, 2.0, 3.0];

        let result = HankelMatrix::from_sequence(sequence.view(), 2, 3); // Need 2+3-1=4 elements
        assert!(result.is_err());
    }
}
