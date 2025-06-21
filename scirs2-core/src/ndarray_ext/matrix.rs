//! Matrix operations for ndarray
//!
//! This module provides matrix creation and manipulation operations similar to
//! those found in `NumPy`/SciPy, such as identity, diagonal, block, and other
//! specialized matrix operations.

use ndarray::{Array, ArrayView, Ix1, Ix2};
use num_traits::{One, Zero};

/// Create an identity matrix
///
/// # Arguments
///
/// * `n` - Number of rows and columns in the square identity matrix
///
/// # Returns
///
/// An nxn identity matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::matrix::eye;
///
/// let id3 = eye::<f64>(3);
/// assert_eq!(id3.shape(), &[3, 3]);
/// assert_eq!(id3[[0, 0]], 1.0);
/// assert_eq!(id3[[1, 1]], 1.0);
/// assert_eq!(id3[[2, 2]], 1.0);
/// assert_eq!(id3[[0, 1]], 0.0);
/// ```
pub fn eye<T>(n: usize) -> Array<T, Ix2>
where
    T: Clone + Zero + One,
{
    let mut result = Array::<T, Ix2>::zeros((n, n));

    for i in 0..n {
        result[[i, i]] = T::one();
    }

    result
}

/// Create a matrix with ones on the given diagonal and zeros elsewhere
///
/// # Arguments
///
/// * `n` - Number of rows
/// * `m` - Number of columns
/// * `k` - Diagonal offset (0 for main diagonal, positive for above, negative for below)
///
/// # Returns
///
/// An n x m matrix with ones on the specified diagonal
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::matrix::eye_offset;
///
/// // 3x3 matrix with ones on the main diagonal (k=0)
/// let id3 = eye_offset::<f64>(3, 3, 0);
/// assert_eq!(id3.shape(), &[3, 3]);
/// assert_eq!(id3[[0, 0]], 1.0);
/// assert_eq!(id3[[1, 1]], 1.0);
/// assert_eq!(id3[[2, 2]], 1.0);
///
/// // 3x4 matrix with ones on the first superdiagonal (k=1)
/// let super_diag = eye_offset::<f64>(3, 4, 1);
/// assert_eq!(super_diag.shape(), &[3, 4]);
/// assert_eq!(super_diag[[0, 1]], 1.0);
/// assert_eq!(super_diag[[1, 2]], 1.0);
/// assert_eq!(super_diag[[2, 3]], 1.0);
/// ```
pub fn eye_offset<T>(n: usize, m: usize, k: isize) -> Array<T, Ix2>
where
    T: Clone + Zero + One,
{
    let mut result = Array::<T, Ix2>::zeros((n, m));

    // Determine the start and end points for the diagonal
    let start_i = if k > 0 { 0 } else { (-k) as usize };
    let start_j = if k < 0 { 0 } else { k as usize };

    let diag_len = std::cmp::min(n - start_i, m - start_j);

    for d in 0..diag_len {
        result[[start_i + d, start_j + d]] = T::one();
    }

    result
}

/// Construct a diagonal matrix from a 1D array
///
/// # Arguments
///
/// * `diag_values` - The values to place on the diagonal
///
/// # Returns
///
/// A square matrix with the input values on the main diagonal and zeros elsewhere
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::matrix::diag;
///
/// let values = array![1, 2, 3];
/// let diag_matrix = diag(values.view());
/// assert_eq!(diag_matrix.shape(), &[3, 3]);
/// assert_eq!(diag_matrix[[0, 0]], 1);
/// assert_eq!(diag_matrix[[1, 1]], 2);
/// assert_eq!(diag_matrix[[2, 2]], 3);
/// assert_eq!(diag_matrix[[0, 1]], 0);
/// ```
pub fn diag<T>(diag_values: ArrayView<T, Ix1>) -> Array<T, Ix2>
where
    T: Clone + Zero,
{
    let n = diag_values.len();
    let mut result = Array::<T, Ix2>::zeros((n, n));

    for i in 0..n {
        result[[i, i]] = diag_values[i].clone();
    }

    result
}

/// Extract a diagonal from a 2D array
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `k` - Diagonal offset (0 for main diagonal, positive for above, negative for below)
///
/// # Returns
///
/// A 1D array containing the specified diagonal
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::matrix::diagonal;
///
/// let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
///
/// // Extract main diagonal
/// let main_diag = diagonal(a.view(), 0).unwrap();
/// assert_eq!(main_diag, array![1, 5, 9]);
///
/// // Extract superdiagonal
/// let super_diag = diagonal(a.view(), 1).unwrap();
/// assert_eq!(super_diag, array![2, 6]);
///
/// // Extract subdiagonal
/// let sub_diag = diagonal(a.view(), -1).unwrap();
/// assert_eq!(sub_diag, array![4, 8]);
/// ```
pub fn diagonal<T>(array: ArrayView<T, Ix2>, k: isize) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Zero,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    // Calculate the length of the diagonal
    let diag_len = if k >= 0 {
        std::cmp::min(rows, cols.saturating_sub(k as usize))
    } else {
        std::cmp::min(cols, rows.saturating_sub((-k) as usize))
    };

    if diag_len == 0 {
        return Err("No diagonal elements for the given offset");
    }

    // Create the result array directly
    let mut result = Array::<T, Ix1>::zeros(diag_len);

    // Extract the diagonal elements
    for i in 0..diag_len {
        let row = if k < 0 { i + (-k) as usize } else { i };

        let col = if k > 0 { i + k as usize } else { i };

        result[i] = array[[row, col]].clone();
    }

    Ok(result)
}

/// Create a matrix filled with a given value
///
/// # Arguments
///
/// * `rows` - Number of rows
/// * `cols` - Number of columns
/// * `value` - Value to fill the matrix with
///
/// # Returns
///
/// A matrix filled with the specified value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::matrix::full;
///
/// let filled = full(2, 3, 7);
/// assert_eq!(filled.shape(), &[2, 3]);
/// assert_eq!(filled[[0, 0]], 7);
/// assert_eq!(filled[[1, 2]], 7);
/// ```
pub fn full<T>(rows: usize, cols: usize, value: T) -> Array<T, Ix2>
where
    T: Clone,
{
    Array::<T, Ix2>::from_elem((rows, cols), value)
}

/// Create a matrix filled with ones
///
/// # Arguments
///
/// * `rows` - Number of rows
/// * `cols` - Number of columns
///
/// # Returns
///
/// A matrix filled with ones
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::matrix::ones;
///
/// let ones_mat = ones::<f64>(2, 3);
/// assert_eq!(ones_mat.shape(), &[2, 3]);
/// assert_eq!(ones_mat[[0, 0]], 1.0);
/// assert_eq!(ones_mat[[1, 2]], 1.0);
/// ```
pub fn ones<T>(rows: usize, cols: usize) -> Array<T, Ix2>
where
    T: Clone + One,
{
    Array::<T, Ix2>::from_elem((rows, cols), T::one())
}

/// Create a matrix filled with zeros
///
/// # Arguments
///
/// * `rows` - Number of rows
/// * `cols` - Number of columns
///
/// # Returns
///
/// A matrix filled with zeros
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::matrix::zeros;
///
/// let zeros_mat = zeros::<f64>(2, 3);
/// assert_eq!(zeros_mat.shape(), &[2, 3]);
/// assert_eq!(zeros_mat[[0, 0]], 0.0);
/// assert_eq!(zeros_mat[[1, 2]], 0.0);
/// ```
pub fn zeros<T>(rows: usize, cols: usize) -> Array<T, Ix2>
where
    T: Clone + Zero,
{
    Array::<T, Ix2>::zeros((rows, cols))
}

/// Compute the Kronecker product of two 2D arrays
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
///
/// # Returns
///
/// The Kronecker product of the input arrays
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::matrix::kron;
///
/// let a = array![[1, 2], [3, 4]];
/// let b = array![[0, 5], [6, 7]];
///
/// let result = kron(a.view(), b.view());
/// assert_eq!(result.shape(), &[4, 4]);
/// assert_eq!(result, array![
///     [0, 5, 0, 10],
///     [6, 7, 12, 14],
///     [0, 15, 0, 20],
///     [18, 21, 24, 28]
/// ]);
/// ```
pub fn kron<T>(a: ArrayView<T, Ix2>, b: ArrayView<T, Ix2>) -> Array<T, Ix2>
where
    T: Clone + Zero + std::ops::Mul<Output = T>,
{
    let (a_rows, a_cols) = (a.shape()[0], a.shape()[1]);
    let (b_rows, b_cols) = (b.shape()[0], b.shape()[1]);

    let result_rows = a_rows * b_rows;
    let result_cols = a_cols * b_cols;

    let mut result = Array::<T, Ix2>::zeros((result_rows, result_cols));

    for i in 0..a_rows {
        for j in 0..a_cols {
            for k in 0..b_rows {
                for l in 0..b_cols {
                    result[[i * b_rows + k, j * b_cols + l]] =
                        a[[i, j]].clone() * b[[k, l]].clone();
                }
            }
        }
    }

    result
}

/// Create a Toeplitz matrix from first row and first column
///
/// # Arguments
///
/// * `first_row` - First row of the Toeplitz matrix
/// * `first_col` - First column of the Toeplitz matrix (first element must match first row's first element)
///
/// # Returns
///
/// A Toeplitz matrix with the specified first row and column
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::matrix::toeplitz;
///
/// let first_row = array![1, 2, 3];
/// let first_col = array![1, 4, 5];
/// let result = toeplitz(first_row.view(), first_col.view()).unwrap();
/// assert_eq!(result.shape(), &[3, 3]);
/// assert_eq!(result, array![
///     [1, 2, 3],
///     [4, 1, 2],
///     [5, 4, 1]
/// ]);
/// ```
pub fn toeplitz<T>(
    first_row: ArrayView<T, Ix1>,
    first_col: ArrayView<T, Ix1>,
) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + PartialEq + Zero,
{
    // First elements of first_row and first_col must match
    if first_row.is_empty() || first_col.is_empty() {
        return Err("Input arrays must not be empty");
    }

    if first_row[0] != first_col[0] {
        return Err("First element of row and column must match");
    }

    let n = first_col.len(); // Number of rows
    let m = first_row.len(); // Number of columns

    let mut result = Array::<T, Ix2>::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            if i <= j {
                // Upper triangle and main diagonal from first_row
                result[[i, j]] = first_row[j - i].clone();
            } else {
                // Lower triangle from first_col
                result[[i, j]] = first_col[i - j].clone();
            }
        }
    }

    Ok(result)
}

/// Create a block diagonal matrix from a sequence of 2D arrays
///
/// # Arguments
///
/// * `arrays` - A slice of 2D arrays to form the blocks on the diagonal
///
/// # Returns
///
/// A block diagonal matrix with the input arrays on the diagonal
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::matrix::block_diag;
///
/// let a = array![[1, 2], [3, 4]];
/// let b = array![[5, 6], [7, 8]];
///
/// let result = block_diag(&[a.view(), b.view()]);
/// assert_eq!(result.shape(), &[4, 4]);
/// assert_eq!(result, array![
///     [1, 2, 0, 0],
///     [3, 4, 0, 0],
///     [0, 0, 5, 6],
///     [0, 0, 7, 8]
/// ]);
/// ```
pub fn block_diag<T>(arrays: &[ArrayView<T, Ix2>]) -> Array<T, Ix2>
where
    T: Clone + Zero,
{
    if arrays.is_empty() {
        return Array::<T, Ix2>::zeros((0, 0));
    }

    // Calculate total dimensions
    let mut total_rows = 0;
    let mut total_cols = 0;

    for array in arrays {
        total_rows += array.shape()[0];
        total_cols += array.shape()[1];
    }

    let mut result = Array::<T, Ix2>::zeros((total_rows, total_cols));

    let mut row_offset = 0;
    let mut col_offset = 0;

    // Place each array on the diagonal
    for array in arrays {
        let (rows, cols) = (array.shape()[0], array.shape()[1]);

        for i in 0..rows {
            for j in 0..cols {
                result[[row_offset + i, col_offset + j]] = array[[i, j]].clone();
            }
        }

        row_offset += rows;
        col_offset += cols;
    }

    result
}

/// Create a tri-diagonal matrix from the three diagonals
///
/// # Arguments
///
/// * `diag` - Main diagonal
/// * `lower_diag` - Lower diagonal
/// * `upper_diag` - Upper diagonal
///
/// # Returns
///
/// A tri-diagonal matrix with the specified diagonals
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::matrix::tridiagonal;
///
/// let diag = array![1, 2, 3];
/// let lower = array![4, 5];
/// let upper = array![6, 7];
///
/// let result = tridiagonal(diag.view(), lower.view(), upper.view()).unwrap();
/// assert_eq!(result.shape(), &[3, 3]);
/// assert_eq!(result, array![
///     [1, 6, 0],
///     [4, 2, 7],
///     [0, 5, 3]
/// ]);
/// ```
pub fn tridiagonal<T>(
    diag: ArrayView<T, Ix1>,
    lower_diag: ArrayView<T, Ix1>,
    upper_diag: ArrayView<T, Ix1>,
) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + Zero,
{
    let n = diag.len();

    // Check that the diagonals have correct sizes
    if lower_diag.len() != n - 1 || upper_diag.len() != n - 1 {
        return Err("Lower and upper diagonals must have length n-1 where n is the length of the main diagonal");
    }

    let mut result = Array::<T, Ix2>::zeros((n, n));

    // Set main diagonal
    for i in 0..n {
        result[[i, i]] = diag[i].clone();
    }

    // Set lower diagonal
    for i in 1..n {
        result[[i, i - 1]] = lower_diag[i - 1].clone();
    }

    // Set upper diagonal
    for i in 0..n - 1 {
        result[[i, i + 1]] = upper_diag[i].clone();
    }

    Ok(result)
}

/// Create a Hankel matrix from its first column and last row
///
/// # Arguments
///
/// * `first_col` - First column of the Hankel matrix
/// * `last_row` - Last row of the Hankel matrix (first element must match last element of first_col)
///
/// # Returns
///
/// A Hankel matrix with the specified first column and last row
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::matrix::hankel;
///
/// let first_col = array![1, 2, 3];
/// let last_row = array![3, 4, 5];
///
/// let result = hankel(first_col.view(), last_row.view()).unwrap();
/// assert_eq!(result.shape(), &[3, 3]);
/// assert_eq!(result, array![
///     [1, 2, 3],
///     [2, 3, 4],
///     [3, 4, 5]
/// ]);
/// ```
pub fn hankel<T>(
    first_col: ArrayView<T, Ix1>,
    last_row: ArrayView<T, Ix1>,
) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + PartialEq + Zero,
{
    if first_col.is_empty() || last_row.is_empty() {
        return Err("Input arrays must not be empty");
    }

    // Last element of first_col must match first element of last_row
    if first_col[first_col.len() - 1] != last_row[0] {
        return Err("Last element of first column must match first element of last row");
    }

    let n = first_col.len(); // Number of rows
    let m = last_row.len(); // Number of columns

    let mut result = Array::<T, Ix2>::zeros((n, m));

    // Combine first_col and last_row (minus first element of last_row) to form the "data" array
    let data_len = n + m - 1;
    let mut data = Vec::with_capacity(data_len);

    // Fill data with first_col elements
    for i in 0..n {
        data.push(first_col[i].clone());
    }

    // Append last_row elements (skipping first element which should match last element of first_col)
    for i in 1..m {
        data.push(last_row[i].clone());
    }

    // Fill the Hankel matrix
    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = data[i + j].clone();
        }
    }

    Ok(result)
}

/// Calculate the trace of a square matrix (sum of diagonal elements)
///
/// # Arguments
///
/// * `array` - Input square matrix
///
/// # Returns
///
/// The trace of the matrix
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::matrix::trace;
///
/// let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
/// let tr = trace(a.view()).unwrap();
/// assert_eq!(tr, 15);  // 1 + 5 + 9 = 15
/// ```
pub fn trace<T>(array: ArrayView<T, Ix2>) -> Result<T, &'static str>
where
    T: Clone + Zero + std::ops::Add<Output = T>,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    if rows != cols {
        return Err("Trace is only defined for square matrices");
    }

    let mut result = T::zero();

    for i in 0..rows {
        result = result + array[[i, i]].clone();
    }

    Ok(result)
}

/// Create a matrix vander from a 1D array
///
/// # Arguments
///
/// * `x` - Input 1D array
/// * `n` - Optional number of columns in the output (defaults to x.len())
/// * `increasing` - Optional boolean to determine order (defaults to false)
///
/// # Returns
///
/// A Vandermonde matrix where each column is a power of the input array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::matrix::vander;
///
/// let x = array![1.0, 2.0, 3.0];
///
/// // Default behavior: decreasing powers from n-1 to 0
/// let v1 = vander(x.view(), None, None).unwrap();
/// assert_eq!(v1.shape(), &[3, 3]);
/// // Powers: x^2, x^1, x^0
/// assert_eq!(v1, array![
///     [1.0, 1.0, 1.0],
///     [4.0, 2.0, 1.0],
///     [9.0, 3.0, 1.0]
/// ]);
///
/// // Increasing powers: 0 to n-1
/// let v2 = vander(x.view(), None, Some(true)).unwrap();
/// assert_eq!(v2.shape(), &[3, 3]);
/// // Powers: x^0, x^1, x^2
/// assert_eq!(v2, array![
///     [1.0, 1.0, 1.0],
///     [1.0, 2.0, 4.0],
///     [1.0, 3.0, 9.0]
/// ]);
///
/// // Specify 4 columns
/// let v3 = vander(x.view(), Some(4), None).unwrap();
/// assert_eq!(v3.shape(), &[3, 4]);
/// // Powers: x^3, x^2, x^1, x^0
/// assert_eq!(v3, array![
///     [1.0, 1.0, 1.0, 1.0],
///     [8.0, 4.0, 2.0, 1.0],
///     [27.0, 9.0, 3.0, 1.0]
/// ]);
/// ```
pub fn vander<T>(
    x: ArrayView<T, Ix1>,
    n: Option<usize>,
    increasing: Option<bool>,
) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + Zero + One + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
{
    let x_len = x.len();

    if x_len == 0 {
        return Err("Input array must not be empty");
    }

    let n = n.unwrap_or(x_len);
    let increasing = increasing.unwrap_or(false);

    let mut result = Array::<T, Ix2>::zeros((x_len, n));

    // Fill in the Vandermonde matrix with powers of x elements
    for i in 0..x_len {
        // Initialize accumulator with 1 (x^0)
        let mut power = T::one();

        if increasing {
            // First column is x^0 (all ones)
            for j in 0..n {
                result[[i, j]] = power.clone();

                if j < n - 1 {
                    power = power.clone() * x[i].clone();
                }
            }
        } else {
            // Decreasing powers (last column is x^0)
            // Calculate highest power first: x^(n-1)
            for _p in 0..n - 1 {
                power = power.clone() * x[i].clone();
            }

            for j in 0..n {
                result[[i, j]] = power.clone();

                if j < n - 1 {
                    // For non-increasing powers, we need Div trait for T to handle division
                    // power = power.clone() / x[i].clone(); // This requires Div trait

                    // Use multiplication by reciprocal instead as a safer approach
                    power = power.clone() * (T::one() / x[i].clone());
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_eye() {
        let id3 = eye::<f64>(3);
        assert_eq!(id3.shape(), &[3, 3]);
        assert_eq!(id3[[0, 0]], 1.0);
        assert_eq!(id3[[1, 1]], 1.0);
        assert_eq!(id3[[2, 2]], 1.0);
        assert_eq!(id3[[0, 1]], 0.0);
        assert_eq!(id3[[1, 0]], 0.0);
    }

    #[test]
    fn test_eye_offset() {
        // 3x3 matrix with ones on the main diagonal (k=0)
        let id3 = eye_offset::<f64>(3, 3, 0);
        assert_eq!(id3.shape(), &[3, 3]);
        assert_eq!(id3[[0, 0]], 1.0);
        assert_eq!(id3[[1, 1]], 1.0);
        assert_eq!(id3[[2, 2]], 1.0);

        // 3x4 matrix with ones on the first superdiagonal (k=1)
        let super_diag = eye_offset::<f64>(3, 4, 1);
        assert_eq!(super_diag.shape(), &[3, 4]);
        assert_eq!(super_diag[[0, 1]], 1.0);
        assert_eq!(super_diag[[1, 2]], 1.0);
        assert_eq!(super_diag[[2, 3]], 1.0);

        // 4x3 matrix with ones on the first subdiagonal (k=-1)
        let sub_diag = eye_offset::<f64>(4, 3, -1);
        assert_eq!(sub_diag.shape(), &[4, 3]);
        assert_eq!(sub_diag[[1, 0]], 1.0);
        assert_eq!(sub_diag[[2, 1]], 1.0);
        assert_eq!(sub_diag[[3, 2]], 1.0);
    }

    #[test]
    fn test_diag() {
        let values = array![1, 2, 3];
        let diag_matrix = diag(values.view());
        assert_eq!(diag_matrix.shape(), &[3, 3]);
        assert_eq!(diag_matrix[[0, 0]], 1);
        assert_eq!(diag_matrix[[1, 1]], 2);
        assert_eq!(diag_matrix[[2, 2]], 3);
        assert_eq!(diag_matrix[[0, 1]], 0);
    }

    #[test]
    fn test_diagonal() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

        // Extract main diagonal
        let main_diag = diagonal(a.view(), 0).unwrap();
        assert_eq!(main_diag, array![1, 5, 9]);

        // Extract superdiagonal
        let super_diag = diagonal(a.view(), 1).unwrap();
        assert_eq!(super_diag, array![2, 6]);

        // Extract subdiagonal
        let sub_diag = diagonal(a.view(), -1).unwrap();
        assert_eq!(sub_diag, array![4, 8]);

        // Test out of bounds (should return error)
        assert!(diagonal(a.view(), 3).is_err());
        assert!(diagonal(a.view(), -3).is_err());
    }

    #[test]
    fn test_full_ones_zeros() {
        // Test full
        let filled = full(2, 3, 7);
        assert_eq!(filled.shape(), &[2, 3]);
        assert_eq!(filled[[0, 0]], 7);
        assert_eq!(filled[[1, 2]], 7);

        // Test ones
        let ones_mat = ones::<f64>(2, 3);
        assert_eq!(ones_mat.shape(), &[2, 3]);
        assert_eq!(ones_mat[[0, 0]], 1.0);
        assert_eq!(ones_mat[[1, 2]], 1.0);

        // Test zeros
        let zeros_mat = zeros::<f64>(2, 3);
        assert_eq!(zeros_mat.shape(), &[2, 3]);
        assert_eq!(zeros_mat[[0, 0]], 0.0);
        assert_eq!(zeros_mat[[1, 2]], 0.0);
    }

    #[test]
    fn test_kron() {
        let a = array![[1, 2], [3, 4]];
        let b = array![[0, 5], [6, 7]];

        let result = kron(a.view(), b.view());
        assert_eq!(result.shape(), &[4, 4]);

        // Check results
        assert_eq!(
            result,
            array![
                [0, 5, 0, 10],
                [6, 7, 12, 14],
                [0, 15, 0, 20],
                [18, 21, 24, 28]
            ]
        );
    }

    #[test]
    fn test_toeplitz() {
        let first_row = array![1, 2, 3];
        let first_col = array![1, 4, 5];

        let result = toeplitz(first_row.view(), first_col.view()).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(result, array![[1, 2, 3], [4, 1, 2], [5, 4, 1]]);

        // Test with mismatched first elements (should return error)
        let bad_row = array![9, 2, 3];
        assert!(toeplitz(bad_row.view(), first_col.view()).is_err());
    }

    #[test]
    fn test_block_diag() {
        let a = array![[1, 2], [3, 4]];
        let b = array![[5, 6], [7, 8]];

        let result = block_diag(&[a.view(), b.view()]);
        assert_eq!(result.shape(), &[4, 4]);
        assert_eq!(
            result,
            array![[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6], [0, 0, 7, 8]]
        );

        // Test with different size blocks
        let c = array![[9]];
        let result2 = block_diag(&[a.view(), c.view()]);
        assert_eq!(result2.shape(), &[3, 3]);
        assert_eq!(result2, array![[1, 2, 0], [3, 4, 0], [0, 0, 9]]);

        // Test with empty array list
        let empty: [ArrayView<i32, Ix2>; 0] = [];
        let result3 = block_diag(&empty);
        assert_eq!(result3.shape(), &[0, 0]);
    }

    #[test]
    fn test_tridiagonal() {
        let diag = array![1, 2, 3];
        let lower = array![4, 5];
        let upper = array![6, 7];

        let result = tridiagonal(diag.view(), lower.view(), upper.view()).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(result, array![[1, 6, 0], [4, 2, 7], [0, 5, 3]]);

        // Test with incorrect diagonals size (should return error)
        let bad_lower = array![4];
        assert!(tridiagonal(diag.view(), bad_lower.view(), upper.view()).is_err());
    }

    #[test]
    fn test_hankel() {
        let first_col = array![1, 2, 3];
        let last_row = array![3, 4, 5];

        let result = hankel(first_col.view(), last_row.view()).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(result, array![[1, 2, 3], [2, 3, 4], [3, 4, 5]]);

        // Test with mismatched elements (should return error)
        let bad_row = array![9, 4, 5];
        assert!(hankel(first_col.view(), bad_row.view()).is_err());
    }

    #[test]
    fn test_trace() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let tr = trace(a.view()).unwrap();
        assert_eq!(tr, 15); // 1 + 5 + 9 = 15

        // Test non-square matrix (should return error)
        let b = array![[1, 2, 3], [4, 5, 6]];
        assert!(trace(b.view()).is_err());
    }

    #[test]
    fn test_vander() {
        let x = array![1.0, 2.0, 3.0];

        // Default behavior: decreasing powers from n-1 to 0
        let v1 = vander(x.view(), None, None).unwrap();
        assert_eq!(v1.shape(), &[3, 3]);
        // Should be equivalent to [x^2, x^1, x^0]
        for i in 0..3 {
            assert_abs_diff_eq!(v1[[i, 0]], x[i] * x[i]);
            assert_abs_diff_eq!(v1[[i, 1]], x[i]);
            assert_abs_diff_eq!(v1[[i, 2]], 1.0);
        }

        // Increasing powers from 0 to n-1
        let v2 = vander(x.view(), None, Some(true)).unwrap();
        assert_eq!(v2.shape(), &[3, 3]);
        // Should be equivalent to [x^0, x^1, x^2]
        for i in 0..3 {
            assert_abs_diff_eq!(v2[[i, 0]], 1.0);
            assert_abs_diff_eq!(v2[[i, 1]], x[i]);
            assert_abs_diff_eq!(v2[[i, 2]], x[i] * x[i]);
        }

        // Specify 4 columns (decreasing power)
        let v3 = vander(x.view(), Some(4), None).unwrap();
        assert_eq!(v3.shape(), &[3, 4]);
        // Should be equivalent to [x^3, x^2, x^1, x^0]
        for i in 0..3 {
            assert_abs_diff_eq!(v3[[i, 0]], x[i] * x[i] * x[i]);
            assert_abs_diff_eq!(v3[[i, 1]], x[i] * x[i]);
            assert_abs_diff_eq!(v3[[i, 2]], x[i]);
            assert_abs_diff_eq!(v3[[i, 3]], 1.0);
        }
    }
}
