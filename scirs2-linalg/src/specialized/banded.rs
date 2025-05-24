//! Banded matrix implementation
//!
//! A banded matrix is a matrix where all non-zero elements are confined to a diagonal band.
//! This structure provides efficient storage and operations for such matrices.

use super::SpecializedMatrix;
use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::{Float, NumAssign, One, Zero};
use std::fmt::Debug;
use std::iter::Sum;

/// Banded matrix representation
///
/// A banded matrix has non-zero elements only within a band around the main diagonal.
/// The band is defined by the number of lower diagonals and upper diagonals.
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_linalg::specialized::BandedMatrix;
/// use scirs2_linalg::SpecializedMatrix;
///
/// // Create a 5x5 banded matrix with lower bandwidth 1 and upper bandwidth 2
/// // This means we have the main diagonal, 1 subdiagonal, and 2 superdiagonals
/// let mut data = Array2::zeros((4, 5)); // 4 diagonals (1+1+2), 5 columns
///
/// // Set the values for each diagonal
/// // Lower diagonal (the 1 subdiagonal)
/// data[[0, 0]] = 1.0;
/// data[[0, 1]] = 2.0;
/// data[[0, 2]] = 3.0;
/// data[[0, 3]] = 4.0;
///
/// // Main diagonal
/// data[[1, 0]] = 5.0;
/// data[[1, 1]] = 6.0;
/// data[[1, 2]] = 7.0;
/// data[[1, 3]] = 8.0;
/// data[[1, 4]] = 9.0;
///
/// // First superdiagonal
/// data[[2, 0]] = 10.0;
/// data[[2, 1]] = 11.0;
/// data[[2, 2]] = 12.0;
/// data[[2, 3]] = 13.0;
///
/// // Second superdiagonal
/// data[[3, 0]] = 14.0;
/// data[[3, 1]] = 15.0;
/// data[[3, 2]] = 16.0;
///
/// let band = BandedMatrix::new(data.view(), 1, 2, 5, 5).unwrap();
///
/// // The matrix is equivalent to:
/// // [[ 5.0, 10.0, 14.0,  0.0,  0.0 ],
/// //  [ 1.0,  6.0, 11.0, 15.0,  0.0 ],
/// //  [ 0.0,  2.0,  7.0, 12.0, 16.0 ],
/// //  [ 0.0,  0.0,  3.0,  8.0, 13.0 ],
/// //  [ 0.0,  0.0,  0.0,  4.0,  9.0 ]]
///
/// // Get elements
/// assert_eq!(band.get(0, 0).unwrap(), 5.0);
/// assert_eq!(band.get(0, 1).unwrap(), 10.0);
/// assert_eq!(band.get(0, 2).unwrap(), 14.0);
/// assert_eq!(band.get(1, 0).unwrap(), 1.0);
/// assert_eq!(band.get(0, 3).unwrap(), 0.0); // Outside the band
///
/// // Matrix-vector multiplication
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = band.matvec(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct BandedMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// The banded data storage
    /// Rows represent diagonals, from the lowest to the highest
    /// data[0] is the lowest diagonal, data[lower_bandwidth + upper_bandwidth] is the highest
    data: Array2<A>,
    /// Number of lower diagonals
    lower_bandwidth: usize,
    /// Number of upper diagonals
    upper_bandwidth: usize,
    /// Number of rows in the matrix
    nrows: usize,
    /// Number of columns in the matrix
    ncols: usize,
}

impl<A> BandedMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// Create a new banded matrix from band storage
    ///
    /// # Arguments
    ///
    /// * `data` - Band storage with diagonals as rows, ordered from lowest to highest
    /// * `lower_bandwidth` - Number of lower diagonals
    /// * `upper_bandwidth` - Number of upper diagonals
    /// * `nrows` - Number of rows in the full matrix
    /// * `ncols` - Number of columns in the full matrix
    ///
    /// # Returns
    ///
    /// * `BandedMatrix` if the data has valid dimensions
    /// * `LinalgError` if dimensions are incompatible
    pub fn new(
        data: ArrayView2<A>,
        lower_bandwidth: usize,
        upper_bandwidth: usize,
        nrows: usize,
        ncols: usize,
    ) -> LinalgResult<Self> {
        // Check dimensions
        let expected_rows = lower_bandwidth + upper_bandwidth + 1;
        if data.nrows() != expected_rows {
            return Err(LinalgError::ShapeError(format!(
                "Data should have {} rows for a matrix with lower bandwidth {} and upper bandwidth {}",
                expected_rows, lower_bandwidth, upper_bandwidth
            )));
        }

        // The maximum length of any diagonal is min(nrows, ncols)
        let max_diag_len = std::cmp::min(nrows, ncols);

        // The number of columns in data should match the maximum possible length
        // of any diagonal in the full matrix
        if data.ncols() != max_diag_len {
            return Err(LinalgError::ShapeError(format!(
                "Data should have {} columns for a matrix with dimensions {}x{}",
                max_diag_len, nrows, ncols
            )));
        }

        Ok(Self {
            data: data.to_owned(),
            lower_bandwidth,
            upper_bandwidth,
            nrows,
            ncols,
        })
    }

    /// Create a new banded matrix from a general matrix, extracting the band
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix
    /// * `lower_bandwidth` - Number of lower diagonals to extract
    /// * `upper_bandwidth` - Number of upper diagonals to extract
    ///
    /// # Returns
    ///
    /// * `BandedMatrix` representation of the input matrix
    pub fn from_matrix(
        a: &ArrayView2<A>,
        lower_bandwidth: usize,
        upper_bandwidth: usize,
    ) -> LinalgResult<Self> {
        let nrows = a.nrows();
        let ncols = a.ncols();

        // The maximum length of any diagonal in the band
        let max_diag_len = std::cmp::min(nrows, ncols);

        // Create band storage
        let mut data = Array2::zeros((lower_bandwidth + upper_bandwidth + 1, max_diag_len));

        // Fill the band storage
        for i in 0..nrows {
            // For each row, extract the elements in the band
            for j in 0..ncols {
                // Check if (i, j) is in the band
                if j < i + upper_bandwidth + 1 && i < j + lower_bandwidth + 1 {
                    // Map (i, j) to the band storage
                    let diag_index = (j as isize - i as isize + lower_bandwidth as isize) as usize;

                    // The position along the diagonal
                    let diag_pos = if j >= i {
                        // Upper diagonal or main diagonal
                        i
                    } else {
                        // Lower diagonal
                        j
                    };

                    data[[diag_index, diag_pos]] = a[[i, j]];
                }
            }
        }

        Ok(Self {
            data,
            lower_bandwidth,
            upper_bandwidth,
            nrows,
            ncols,
        })
    }

    /// Get the bandwidth of the matrix (sum of lower and upper bandwidth plus one)
    pub fn bandwidth(&self) -> usize {
        self.lower_bandwidth + self.upper_bandwidth + 1
    }

    /// Get the lower bandwidth of the matrix
    pub fn lower_bandwidth(&self) -> usize {
        self.lower_bandwidth
    }

    /// Get the upper bandwidth of the matrix
    pub fn upper_bandwidth(&self) -> usize {
        self.upper_bandwidth
    }

    /// Solve a banded system of equations Ax = b
    ///
    /// This implementation uses a modified LU decomposition for banded matrices.
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand side vector
    ///
    /// # Returns
    ///
    /// * Solution vector x
    /// * `LinalgError` if the matrix is singular or dimensions are incompatible
    pub fn solve(&self, b: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if b.len() != self.nrows {
            return Err(LinalgError::ShapeError(format!(
                "Right-hand side length {} does not match matrix dimension {}",
                b.len(),
                self.nrows
            )));
        }

        if self.nrows != self.ncols {
            return Err(LinalgError::ShapeError(
                "Matrix must be square to solve system".to_string(),
            ));
        }

        // For very simple systems, use direct solution
        if self.nrows == 1 {
            let a = self.get(0, 0)?;
            if a.abs() < A::epsilon() {
                return Err(LinalgError::SingularMatrixError(
                    "Matrix is singular".to_string(),
                ));
            }

            let mut x = Array1::zeros(1);
            x[0] = b[0] / a;
            return Ok(x);
        }

        // For tridiagonal systems (lower_bandwidth=1, upper_bandwidth=1), use a specialized algorithm
        if self.lower_bandwidth == 1 && self.upper_bandwidth == 1 {
            return self.solve_tridiagonal(b);
        }

        // For general banded systems, use a fallback approach
        // This is temporary and should be replaced with a proper banded LU decomposition
        let mut x = Array1::zeros(self.nrows);

        // Extract to dense and use a naive Gaussian elimination
        let a_dense = self.to_dense()?;
        let mut augmented = Array2::zeros((self.nrows, self.ncols + 1));

        // Copy A into the augmented matrix
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                augmented[[i, j]] = a_dense[[i, j]];
            }
            augmented[[i, self.ncols]] = b[i];
        }

        // Gaussian elimination
        for i in 0..self.nrows - 1 {
            // Find pivot
            let mut max_row = i;
            let mut max_val = augmented[[i, i]].abs();

            for j in i + 1..self.nrows {
                let val = augmented[[j, i]].abs();
                if val > max_val {
                    max_val = val;
                    max_row = j;
                }
            }

            // Check for singularity
            if max_val < A::epsilon() {
                return Err(LinalgError::SingularMatrixError(
                    "Matrix is singular during Gaussian elimination".to_string(),
                ));
            }

            // Swap rows if needed
            if max_row != i {
                for j in i..=self.ncols {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Eliminate
            for j in i + 1..self.nrows {
                let factor = augmented[[j, i]] / augmented[[i, i]];

                for k in i..=self.ncols {
                    let value_i_k = augmented[[i, k]];
                    augmented[[j, k]] -= factor * value_i_k;
                }
            }
        }

        // Back substitution
        for i in (0..self.nrows).rev() {
            let mut sum = A::zero();

            for j in i + 1..self.ncols {
                sum += augmented[[i, j]] * x[j];
            }

            x[i] = (augmented[[i, self.ncols]] - sum) / augmented[[i, i]];
        }

        Ok(x)
    }

    /// Solve a tridiagonal system Ax = b using the Thomas algorithm
    ///
    /// This is a specialized solver for tridiagonal matrices (lower_bandwidth=1, upper_bandwidth=1)
    fn solve_tridiagonal(&self, b: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        let n = self.nrows;

        // Check dimensions
        if n != b.len() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix rows ({}) must match vector length ({})",
                n,
                b.len()
            )));
        }

        // Check if the matrix is actually tridiagonal
        if self.lower_bandwidth != 1 || self.upper_bandwidth != 1 {
            return Err(LinalgError::ShapeError(
                "solve_tridiagonal requires a matrix with lower_bandwidth=1 and upper_bandwidth=1"
                    .to_string(),
            ));
        }

        // Extract the diagonals for easier access
        let mut lower = Array1::zeros(n - 1); // Subdiagonal
        let mut diag = Array1::zeros(n); // Main diagonal
        let mut upper = Array1::zeros(n - 1); // Superdiagonal

        // Extract the diagonals from band storage
        for i in 0..n {
            // Main diagonal
            diag[i] = self.get(i, i)?;

            // Subdiagonal (lower)
            if i > 0 {
                lower[i - 1] = self.get(i, i - 1)?;
            }

            // Superdiagonal (upper)
            if i < n - 1 {
                upper[i] = self.get(i, i + 1)?;
            }
        }

        // Allocate arrays for the solution
        let mut c_prime = Array1::zeros(n - 1);
        let mut d_prime = Array1::zeros(n);
        let mut x = Array1::zeros(n);

        // Forward sweep (elimination)
        if diag[0].abs() < A::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular during tridiagonal solve: zero on main diagonal".to_string(),
            ));
        }

        d_prime[0] = b[0] / diag[0];
        c_prime[0] = upper[0] / diag[0];

        for i in 1..n - 1 {
            let m = lower[i - 1] / diag[i - 1];
            let new_diag = diag[i] - m * upper[i - 1];

            if new_diag.abs() < A::epsilon() {
                return Err(LinalgError::SingularMatrixError(
                    "Matrix is singular during tridiagonal solve: zero pivot encountered"
                        .to_string(),
                ));
            }

            d_prime[i] = (b[i] - lower[i - 1] * d_prime[i - 1]) / new_diag;
            c_prime[i] = upper[i] / new_diag;

            // Update the diagonal for the next iteration
            diag[i] = new_diag;
        }

        // Handle the last row
        if n > 1 {
            let m = lower[n - 2] / diag[n - 2];
            let new_diag = diag[n - 1] - m * upper[n - 2];

            if new_diag.abs() < A::epsilon() {
                return Err(LinalgError::SingularMatrixError(
                    "Matrix is singular during tridiagonal solve: zero pivot in last row"
                        .to_string(),
                ));
            }

            d_prime[n - 1] = (b[n - 1] - lower[n - 2] * d_prime[n - 2]) / new_diag;
        }

        // Back substitution
        x[n - 1] = d_prime[n - 1];

        for i in (0..n - 1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }

        Ok(x)
    }
}

impl<A> SpecializedMatrix<A> for BandedMatrix<A>
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
                "Index ({}, {}) out of bounds for matrix of size {}x{}",
                i, j, self.nrows, self.ncols
            )));
        }

        // Check if the element is within the band
        if j > i + self.upper_bandwidth || i > j + self.lower_bandwidth {
            return Ok(A::zero());
        }

        // Map (i, j) to the band storage
        let diag_index = (j as isize - i as isize + self.lower_bandwidth as isize) as usize;

        // The position along the diagonal
        let diag_pos = if j >= i {
            // Upper diagonal or main diagonal
            i
        } else {
            // Lower diagonal
            j
        };

        Ok(self.data[[diag_index, diag_pos]])
    }

    fn matvec(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if x.len() != self.ncols {
            return Err(LinalgError::ShapeError(format!(
                "Vector length {} does not match matrix column count {}",
                x.len(),
                self.ncols
            )));
        }

        let mut y = Array1::zeros(self.nrows);

        // Compute y = Ax for a banded matrix
        for i in 0..self.nrows {
            // For each row, compute the dot product with x
            // within the band

            // Determine the column range for this row
            let j_start = i.saturating_sub(self.lower_bandwidth);
            let j_end = std::cmp::min(i + self.upper_bandwidth + 1, self.ncols);

            for j in j_start..j_end {
                // Map to band storage
                let diag_index = (j as isize - i as isize + self.lower_bandwidth as isize) as usize;

                // The position along the diagonal
                let diag_pos = if j >= i {
                    // Upper diagonal or main diagonal
                    i
                } else {
                    // Lower diagonal
                    j
                };

                y[i] += self.data[[diag_index, diag_pos]] * x[j];
            }
        }

        Ok(y)
    }

    fn matvec_transpose(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if x.len() != self.nrows {
            return Err(LinalgError::ShapeError(format!(
                "Vector length {} does not match matrix row count {}",
                x.len(),
                self.nrows
            )));
        }

        let mut y = Array1::zeros(self.ncols);

        // Compute y = A^T x for a banded matrix
        // For each column j of A (which becomes row j of A^T)
        for j in 0..self.ncols {
            // Determine the row range for this column
            let i_start = j.saturating_sub(self.upper_bandwidth);
            let i_end = std::cmp::min(j + self.lower_bandwidth + 1, self.nrows);

            for i in i_start..i_end {
                // Map to band storage
                let diag_index = (j as isize - i as isize + self.lower_bandwidth as isize) as usize;

                // The position along the diagonal
                let diag_pos = if j >= i {
                    // Upper diagonal or main diagonal
                    i
                } else {
                    // Lower diagonal
                    j
                };

                y[j] += self.data[[diag_index, diag_pos]] * x[i];
            }
        }

        Ok(y)
    }

    fn to_dense(&self) -> LinalgResult<Array2<A>> {
        let mut a = Array2::zeros((self.nrows, self.ncols));

        for i in 0..self.nrows {
            // Determine the column range for this row
            let j_start = i.saturating_sub(self.lower_bandwidth);
            let j_end = std::cmp::min(i + self.upper_bandwidth + 1, self.ncols);

            for j in j_start..j_end {
                // Map to band storage
                let diag_index = (j as isize - i as isize + self.lower_bandwidth as isize) as usize;

                // The position along the diagonal
                let diag_pos = if j >= i {
                    // Upper diagonal or main diagonal
                    i
                } else {
                    // Lower diagonal
                    j
                };

                a[[i, j]] = self.data[[diag_index, diag_pos]];
            }
        }

        Ok(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_banded_creation() {
        // Create a 5x5 banded matrix with lower bandwidth 1 and upper bandwidth 2
        let mut data = Array2::zeros((4, 5)); // 4 diagonals (1+1+2), 5 columns

        // Set the values for each diagonal
        // Lower diagonal (the 1 subdiagonal)
        data[[0, 0]] = 1.0;
        data[[0, 1]] = 2.0;
        data[[0, 2]] = 3.0;
        data[[0, 3]] = 4.0;

        // Main diagonal
        data[[1, 0]] = 5.0;
        data[[1, 1]] = 6.0;
        data[[1, 2]] = 7.0;
        data[[1, 3]] = 8.0;
        data[[1, 4]] = 9.0;

        // First superdiagonal
        data[[2, 0]] = 10.0;
        data[[2, 1]] = 11.0;
        data[[2, 2]] = 12.0;
        data[[2, 3]] = 13.0;

        // Second superdiagonal
        data[[3, 0]] = 14.0;
        data[[3, 1]] = 15.0;
        data[[3, 2]] = 16.0;

        let band = BandedMatrix::new(data.view(), 1, 2, 5, 5).unwrap();

        assert_eq!(band.nrows(), 5);
        assert_eq!(band.ncols(), 5);
        assert_eq!(band.bandwidth(), 4);
        assert_eq!(band.lower_bandwidth(), 1);
        assert_eq!(band.upper_bandwidth(), 2);

        // Check elements
        assert_relative_eq!(band.get(0, 0).unwrap(), 5.0);
        assert_relative_eq!(band.get(0, 1).unwrap(), 10.0);
        assert_relative_eq!(band.get(0, 2).unwrap(), 14.0);
        assert_relative_eq!(band.get(1, 0).unwrap(), 1.0);
        assert_relative_eq!(band.get(1, 1).unwrap(), 6.0);
        assert_relative_eq!(band.get(1, 2).unwrap(), 11.0);
        assert_relative_eq!(band.get(1, 3).unwrap(), 15.0);

        // Check zero elements outside the band
        assert_relative_eq!(band.get(0, 3).unwrap(), 0.0);
        assert_relative_eq!(band.get(0, 4).unwrap(), 0.0);
        assert_relative_eq!(band.get(3, 0).unwrap(), 0.0);
    }

    #[test]
    fn test_from_matrix() {
        // Create a dense matrix
        let a = array![
            [5.0, 10.0, 14.0, 0.0, 0.0],
            [1.0, 6.0, 11.0, 15.0, 0.0],
            [0.0, 2.0, 7.0, 12.0, 16.0],
            [0.0, 0.0, 3.0, 8.0, 13.0],
            [0.0, 0.0, 0.0, 4.0, 9.0]
        ];

        // Convert to banded form
        let band = BandedMatrix::from_matrix(&a.view(), 1, 2).unwrap();

        assert_eq!(band.nrows(), 5);
        assert_eq!(band.ncols(), 5);
        assert_eq!(band.bandwidth(), 4);
        assert_eq!(band.lower_bandwidth(), 1);
        assert_eq!(band.upper_bandwidth(), 2);

        // Check elements
        assert_relative_eq!(band.get(0, 0).unwrap(), 5.0);
        assert_relative_eq!(band.get(0, 1).unwrap(), 10.0);
        assert_relative_eq!(band.get(0, 2).unwrap(), 14.0);
        assert_relative_eq!(band.get(1, 0).unwrap(), 1.0);
        assert_relative_eq!(band.get(1, 1).unwrap(), 6.0);
        assert_relative_eq!(band.get(1, 2).unwrap(), 11.0);
        assert_relative_eq!(band.get(1, 3).unwrap(), 15.0);

        // Check zero elements outside the band
        assert_relative_eq!(band.get(0, 3).unwrap(), 0.0);
        assert_relative_eq!(band.get(0, 4).unwrap(), 0.0);
        assert_relative_eq!(band.get(3, 0).unwrap(), 0.0);
    }

    #[test]
    fn test_matvec() {
        // Create a banded matrix
        let mut data = Array2::zeros((3, 4)); // 3 diagonals (1+1+1), 4 columns

        // Lower diagonal
        data[[0, 0]] = 1.0;
        data[[0, 1]] = 2.0;
        data[[0, 2]] = 3.0;

        // Main diagonal
        data[[1, 0]] = 4.0;
        data[[1, 1]] = 5.0;
        data[[1, 2]] = 6.0;
        data[[1, 3]] = 7.0;

        // Upper diagonal
        data[[2, 0]] = 8.0;
        data[[2, 1]] = 9.0;
        data[[2, 2]] = 10.0;

        let band = BandedMatrix::new(data.view(), 1, 1, 4, 4).unwrap();

        // Matrix is equivalent to:
        // [[ 4.0, 8.0, 0.0, 0.0 ],
        //  [ 1.0, 5.0, 9.0, 0.0 ],
        //  [ 0.0, 2.0, 6.0, 10.0 ],
        //  [ 0.0, 0.0, 3.0, 7.0 ]]

        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = band.matvec(&x.view()).unwrap();

        // Expected: y = A * x
        let expected = array![
            4.0 * 1.0 + 8.0 * 2.0,
            1.0 * 1.0 + 5.0 * 2.0 + 9.0 * 3.0,
            2.0 * 2.0 + 6.0 * 3.0 + 10.0 * 4.0,
            3.0 * 3.0 + 7.0 * 4.0
        ];

        assert_eq!(y.len(), 4);
        assert_relative_eq!(y[0], expected[0], epsilon = 1e-10);
        assert_relative_eq!(y[1], expected[1], epsilon = 1e-10);
        assert_relative_eq!(y[2], expected[2], epsilon = 1e-10);
        assert_relative_eq!(y[3], expected[3], epsilon = 1e-10);
    }

    #[test]
    fn test_matvec_transpose() {
        // Create a banded matrix
        let mut data = Array2::zeros((3, 4)); // 3 diagonals (1+1+1), 4 columns

        // Lower diagonal
        data[[0, 0]] = 1.0;
        data[[0, 1]] = 2.0;
        data[[0, 2]] = 3.0;

        // Main diagonal
        data[[1, 0]] = 4.0;
        data[[1, 1]] = 5.0;
        data[[1, 2]] = 6.0;
        data[[1, 3]] = 7.0;

        // Upper diagonal
        data[[2, 0]] = 8.0;
        data[[2, 1]] = 9.0;
        data[[2, 2]] = 10.0;

        let band = BandedMatrix::new(data.view(), 1, 1, 4, 4).unwrap();

        // Matrix is equivalent to:
        // [[ 4.0, 8.0, 0.0, 0.0 ],
        //  [ 1.0, 5.0, 9.0, 0.0 ],
        //  [ 0.0, 2.0, 6.0, 10.0 ],
        //  [ 0.0, 0.0, 3.0, 7.0 ]]

        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = band.matvec_transpose(&x.view()).unwrap();

        // Expected: y = A^T * x
        let expected = array![
            4.0 * 1.0 + 1.0 * 2.0,
            8.0 * 1.0 + 5.0 * 2.0 + 2.0 * 3.0,
            9.0 * 2.0 + 6.0 * 3.0 + 3.0 * 4.0,
            10.0 * 3.0 + 7.0 * 4.0
        ];

        assert_eq!(y.len(), 4);
        assert_relative_eq!(y[0], expected[0], epsilon = 1e-10);
        assert_relative_eq!(y[1], expected[1], epsilon = 1e-10);
        assert_relative_eq!(y[2], expected[2], epsilon = 1e-10);
        assert_relative_eq!(y[3], expected[3], epsilon = 1e-10);
    }

    #[test]
    fn test_to_dense() {
        // Create a banded matrix
        let mut data = Array2::zeros((3, 3)); // 3 diagonals (1+1+1), 3 columns

        // Lower diagonal
        data[[0, 0]] = 1.0;
        data[[0, 1]] = 2.0;

        // Main diagonal
        data[[1, 0]] = 3.0;
        data[[1, 1]] = 4.0;
        data[[1, 2]] = 5.0;

        // Upper diagonal
        data[[2, 0]] = 6.0;
        data[[2, 1]] = 7.0;

        let band = BandedMatrix::new(data.view(), 1, 1, 3, 3).unwrap();

        let dense = band.to_dense().unwrap();

        let expected = array![[3.0, 6.0, 0.0], [1.0, 4.0, 7.0], [0.0, 2.0, 5.0]];

        assert_eq!(dense.shape(), &[3, 3]);

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(dense[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_solve() {
        // Create a simple tridiagonal system:
        // [2 -1  0] [x0]   [1]
        // [-1 2 -1] [x1] = [2]
        // [0 -1  2] [x2]   [3]

        // Create the original dense matrix
        let a = array![[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]];

        // Convert to banded form
        let band = BandedMatrix::from_matrix(&a.view(), 1, 1).unwrap();

        // Right-hand side b = [1, 2, 3]
        let b = array![1.0, 2.0, 3.0];

        // The analytical solution is x = [2.5, 4.0, 3.5]
        let expected = array![2.5, 4.0, 3.5];

        // Solve with our tridiagonal solver
        let x = band.solve_tridiagonal(&b.view()).unwrap();

        assert_eq!(x.len(), 3);
        assert_relative_eq!(x[0], expected[0], epsilon = 1e-10);
        assert_relative_eq!(x[1], expected[1], epsilon = 1e-10);
        assert_relative_eq!(x[2], expected[2], epsilon = 1e-10);

        // Double-check by verifying that Ax ≈ b
        let ax = band.matvec(&x.view()).unwrap();

        assert_eq!(ax.len(), 3);
        assert_relative_eq!(ax[0], b[0], epsilon = 1e-10);
        assert_relative_eq!(ax[1], b[1], epsilon = 1e-10);
        assert_relative_eq!(ax[2], b[2], epsilon = 1e-10);

        // Also test the main solve function
        let x2 = band.solve(&b.view()).unwrap();

        assert_eq!(x2.len(), 3);
        assert_relative_eq!(x2[0], expected[0], epsilon = 1e-10);
        assert_relative_eq!(x2[1], expected[1], epsilon = 1e-10);
        assert_relative_eq!(x2[2], expected[2], epsilon = 1e-10);
    }
}
