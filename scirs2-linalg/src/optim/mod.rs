//! Optimized matrix operations for large parameter matrices
//!
//! This module provides efficient implementations of matrix operations
//! that are specifically optimized for large parameter matrices in
//! machine learning applications.

use scirs2_core::ndarray::{s, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

/// Block-based matrix multiplication optimized for large matrices
///
/// Performs matrix multiplication using a block-based approach that optimizes
/// cache usage for large matrices, which is common in neural network models
/// with many parameters.
///
/// # Arguments
///
/// * `a` - First matrix of shape (m, k)
/// * `b` - Second matrix of shape (k, n)
/// * `blocksize` - Size of blocks to use (default: 64)
///
/// # Returns
///
/// * Result matrix of shape (m, n)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::optim::block_matmul;
///
/// // Create two matrices
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3
/// let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]; // 3x2
///
/// // Perform block-based matrix multiplication
/// let result = block_matmul(&a.view(), &b.view(), None).unwrap();
///
/// // Verify the result
/// assert_eq!(result.shape(), &[2, 2]);
/// assert_eq!(result[[0, 0]], 58.0); // 1*7 + 2*9 + 3*11
/// assert_eq!(result[[0, 1]], 64.0); // 1*8 + 2*10 + 3*12
/// assert_eq!(result[[1, 0]], 139.0); // 4*7 + 5*9 + 6*11
/// assert_eq!(result[[1, 1]], 154.0); // 4*8 + 5*10 + 6*12
/// ```
#[allow(dead_code)]
pub fn block_matmul<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    blocksize: Option<usize>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    // Check dimensions compatibility
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Inner dimensions mismatch for matmul: {k1} vs {k2}"
        )));
    }

    // Use the provided block size or default to 64
    let blocksize = blocksize.unwrap_or(64);

    // Initialize result matrix
    let mut result = Array2::zeros((m, n));

    // Compute the number of blocks in each dimension
    let m_blocks = m.div_ceil(blocksize);
    let n_blocks = n.div_ceil(blocksize);
    let k_blocks = k1.div_ceil(blocksize);

    // Perform block-based matrix multiplication
    for mb in 0..m_blocks {
        let m_start = mb * blocksize;
        let m_end = (m_start + blocksize).min(m);

        for nb in 0..n_blocks {
            let n_start = nb * blocksize;
            let n_end = (n_start + blocksize).min(n);

            // Initialize block result to zero
            for i in m_start..m_end {
                for j in n_start..n_end {
                    result[[i, j]] = F::zero();
                }
            }

            // Accumulate over k-dimension blocks
            for kb in 0..k_blocks {
                let k_start = kb * blocksize;
                let k_end = (k_start + blocksize).min(k1);

                // Compute the block multiplication
                for i in m_start..m_end {
                    for j in n_start..n_end {
                        for k in k_start..k_end {
                            result[[i, j]] += a[[i, k]] * b[[k, j]];
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Strassen algorithm for matrix multiplication
///
/// Implements the Strassen algorithm for matrix multiplication, which has a
/// lower asymptotic complexity (O(n^2.807)) compared to the standard algorithm (O(n^3)).
/// This is particularly beneficial for large matrices.
///
/// Note: For optimal performance, the input matrices should have dimensions
/// that are powers of 2. Otherwise, they will be padded internally.
///
/// # Arguments
///
/// * `a` - First matrix
/// * `b` - Second matrix
/// * `cutoff` - Threshold size below which to use standard multiplication (default: 128)
///
/// # Returns
///
/// * Result matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::optim::strassen_matmul;
///
/// // Create two matrices
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[5.0, 6.0], [7.0, 8.0]];
///
/// // Perform Strassen matrix multiplication
/// let result = strassen_matmul(&a.view(), &b.view(), None).unwrap();
///
/// // Verify the result
/// assert_eq!(result.shape(), &[2, 2]);
/// assert_eq!(result[[0, 0]], 19.0); // 1*5 + 2*7
/// assert_eq!(result[[0, 1]], 22.0); // 1*6 + 2*8
/// assert_eq!(result[[1, 0]], 43.0); // 3*5 + 4*7
/// assert_eq!(result[[1, 1]], 50.0); // 3*6 + 4*8
/// ```
#[allow(dead_code)]
pub fn strassen_matmul<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    cutoff: Option<usize>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    // Check dimensions compatibility
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Inner dimensions mismatch for matmul: {k1} vs {k2}"
        )));
    }

    // Use the provided cutoff or default to 128
    let cutoff = cutoff.unwrap_or(128);

    // If the matrices are small, use standard matrix multiplication
    if m <= cutoff || k1 <= cutoff || n <= cutoff {
        return standard_matmul(a, b);
    }

    // Find the next power of 2 for each dimension
    let newsize = 1 << ((m.max(k1).max(n) - 1).ilog2() + 1);

    // Pad matrices to power of 2 dimensions
    let a_padded = padmatrix(a, newsize, newsize);
    let b_padded = padmatrix(b, newsize, newsize);

    // Recursively compute using Strassen algorithm
    let result_padded = strassen_recursive(&a_padded.view(), &b_padded.view(), cutoff);

    // Extract the result with the original dimensions
    let result = result_padded.slice(s![0..m, 0..n]).to_owned();

    Ok(result)
}

/// Pad a matrix to the specified dimensions
#[allow(dead_code)]
fn padmatrix<F>(a: &ArrayView2<F>, new_rows: usize, newcols: usize) -> Array2<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (rows, cols) = a.dim();
    let mut result = Array2::zeros((new_rows, newcols));

    // Copy the original matrix
    for i in 0..rows {
        for j in 0..cols {
            result[[i, j]] = a[[i, j]];
        }
    }

    result
}

/// Internal recursive implementation of the Strassen algorithm
#[allow(dead_code)]
fn strassen_recursive<F>(a: &ArrayView2<F>, b: &ArrayView2<F>, cutoff: usize) -> Array2<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand,
{
    let n = a.dim().0; // Assume square matrices after padding

    // Base case: use standard algorithm for small matrices
    if n <= cutoff {
        return standard_matmul(a, b).unwrap();
    }

    let half_n = n / 2;

    // Divide matrices into quadrants
    let a11 = a.slice(s![0..half_n, 0..half_n]);
    let a12 = a.slice(s![0..half_n, half_n..n]);
    let a21 = a.slice(s![half_n..n, 0..half_n]);
    let a22 = a.slice(s![half_n..n, half_n..n]);

    let b11 = b.slice(s![0..half_n, 0..half_n]);
    let b12 = b.slice(s![0..half_n, half_n..n]);
    let b21 = b.slice(s![half_n..n, 0..half_n]);
    let b22 = b.slice(s![half_n..n, half_n..n]);

    // Compute Strassen's seven products
    let p1 = strassen_recursive(&(&a11 + &a22).view(), &(&b11 + &b22).view(), cutoff);
    let p2 = strassen_recursive(&(&a21 + &a22).view(), &b11.view(), cutoff);
    let p3 = strassen_recursive(&a11.view(), &(&b12 - &b22).view(), cutoff);
    let p4 = strassen_recursive(&a22.view(), &(&b21 - &b11).view(), cutoff);
    let p5 = strassen_recursive(&(&a11 + &a12).view(), &b22.view(), cutoff);
    let p6 = strassen_recursive(&(&a21 - &a11).view(), &(&b11 + &b12).view(), cutoff);
    let p7 = strassen_recursive(&(&a12 - &a22).view(), &(&b21 + &b22).view(), cutoff);

    // Compute the quadrants of the result
    let c11 = &p1 + &p4 - &p5 + &p7;
    let c12 = &p3 + &p5;
    let c21 = &p2 + &p4;
    let c22 = &p1 - &p2 + &p3 + &p6;

    // Combine the quadrants into the result matrix
    let mut result = Array2::zeros((n, n));
    for i in 0..half_n {
        for j in 0..half_n {
            result[[i, j]] = c11[[i, j]];
            result[[i, j + half_n]] = c12[[i, j]];
            result[[i + half_n, j]] = c21[[i, j]];
            result[[i + half_n, j + half_n]] = c22[[i, j]];
        }
    }

    result
}

/// Standard matrix multiplication implementation
#[allow(dead_code)]
fn standard_matmul<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    // Check dimensions compatibility
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Inner dimensions mismatch for matmul: {k1} vs {k2}"
        )));
    }

    // Initialize result matrix
    let mut result = Array2::zeros((m, n));

    // Perform standard matrix multiplication
    for i in 0..m {
        for j in 0..n {
            for k in 0..k1 {
                result[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }

    Ok(result)
}

/// Tiled matrix multiplication optimized for cache efficiency
///
/// Implements a tiled matrix multiplication algorithm that improves
/// cache locality, which is particularly beneficial for large matrices.
///
/// # Arguments
///
/// * `a` - First matrix
/// * `b` - Second matrix
/// * `tilesize` - Size of the tiles (default: 32)
///
/// # Returns
///
/// * Result matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::optim::tiled_matmul;
///
/// // Create two matrices
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3
/// let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]; // 3x2
///
/// // Perform tiled matrix multiplication
/// let result = tiled_matmul(&a.view(), &b.view(), None).unwrap();
///
/// // Verify the result
/// assert_eq!(result.shape(), &[2, 2]);
/// assert_eq!(result[[0, 0]], 58.0); // 1*7 + 2*9 + 3*11
/// assert_eq!(result[[0, 1]], 64.0); // 1*8 + 2*10 + 3*12
/// assert_eq!(result[[1, 0]], 139.0); // 4*7 + 5*9 + 6*11
/// assert_eq!(result[[1, 1]], 154.0); // 4*8 + 5*10 + 6*12
/// ```
#[allow(dead_code)]
pub fn tiled_matmul<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    tilesize: Option<usize>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    // Check dimensions compatibility
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Inner dimensions mismatch for matmul: {k1} vs {k2}"
        )));
    }

    // Use the provided tile size or default to 32
    let tilesize = tilesize.unwrap_or(32);

    // Initialize result matrix
    let mut result = Array2::zeros((m, n));

    // Compute the number of tiles in each dimension
    let m_tiles = m.div_ceil(tilesize);
    let n_tiles = n.div_ceil(tilesize);
    let k_tiles = k1.div_ceil(tilesize);

    // Optimized cache-friendly tiled matrix multiplication
    // Use i -> k -> j loop order for better cache locality
    for i_tile in 0..m_tiles {
        let i_start = i_tile * tilesize;
        let i_end = (i_start + tilesize).min(m);

        for k_tile in 0..k_tiles {
            let k_start = k_tile * tilesize;
            let k_end = (k_start + tilesize).min(k1);

            for j_tile in 0..n_tiles {
                let j_start = j_tile * tilesize;
                let j_end = (j_start + tilesize).min(n);

                // Process the current tile with optimal cache access pattern
                for i in i_start..i_end {
                    for k in k_start..k_end {
                        let a_ik = a[[i, k]];
                        for j in j_start..j_end {
                            result[[i, j]] += a_ik * b[[k, j]];
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_block_matmul_2x2() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result = block_matmul(&a.view(), &b.view(), Some(1)).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_relative_eq!(result[[0, 0]], 19.0);
        assert_relative_eq!(result[[0, 1]], 22.0);
        assert_relative_eq!(result[[1, 0]], 43.0);
        assert_relative_eq!(result[[1, 1]], 50.0);
    }

    #[test]
    fn test_block_matmul_2x3_3x2() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];

        let result = block_matmul(&a.view(), &b.view(), Some(2)).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_relative_eq!(result[[0, 0]], 58.0);
        assert_relative_eq!(result[[0, 1]], 64.0);
        assert_relative_eq!(result[[1, 0]], 139.0);
        assert_relative_eq!(result[[1, 1]], 154.0);
    }

    #[test]
    fn test_strassen_matmul_2x2() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result = strassen_matmul(&a.view(), &b.view(), Some(1)).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_relative_eq!(result[[0, 0]], 19.0);
        assert_relative_eq!(result[[0, 1]], 22.0);
        assert_relative_eq!(result[[1, 0]], 43.0);
        assert_relative_eq!(result[[1, 1]], 50.0);
    }

    #[test]
    fn test_strassen_matmul_3x3() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let b = array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];

        let result = strassen_matmul(&a.view(), &b.view(), Some(2)).unwrap();

        // Compute expected result manually
        let expected = array![[30.0, 24.0, 18.0], [84.0, 69.0, 54.0], [138.0, 114.0, 90.0]];

        assert_eq!(result.shape(), &[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(result[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_tiled_matmul_2x2() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result = tiled_matmul(&a.view(), &b.view(), Some(1)).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_relative_eq!(result[[0, 0]], 19.0);
        assert_relative_eq!(result[[0, 1]], 22.0);
        assert_relative_eq!(result[[1, 0]], 43.0);
        assert_relative_eq!(result[[1, 1]], 50.0);
    }

    #[test]
    fn test_tiled_matmul_2x3_3x2() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];

        let result = tiled_matmul(&a.view(), &b.view(), Some(2)).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_relative_eq!(result[[0, 0]], 58.0);
        assert_relative_eq!(result[[0, 1]], 64.0);
        assert_relative_eq!(result[[1, 0]], 139.0);
        assert_relative_eq!(result[[1, 1]], 154.0);
    }

    #[test]
    fn test_largematrix_equivalence() {
        // Create larger matrices to test the algorithms
        let size = 20;
        let mut a = Array2::<f64>::zeros((size, size));
        let mut b = Array2::<f64>::zeros((size, size));

        // Fill with data
        for i in 0..size {
            for j in 0..size {
                a[[i, j]] = (i * size + j) as f64;
                b[[i, j]] = ((size - i) * size + (size - j)) as f64;
            }
        }

        // Compute results using different algorithms
        let result_standard = standard_matmul(&a.view(), &b.view()).unwrap();
        let result_block = block_matmul(&a.view(), &b.view(), Some(4)).unwrap();
        let result_strassen = strassen_matmul(&a.view(), &b.view(), Some(8)).unwrap();
        let result_tiled = tiled_matmul(&a.view(), &b.view(), Some(4)).unwrap();

        // Compare results
        for i in 0..size {
            for j in 0..size {
                assert_relative_eq!(
                    result_block[[i, j]],
                    result_standard[[i, j]],
                    epsilon = 1e-10
                );
                assert_relative_eq!(
                    result_strassen[[i, j]],
                    result_standard[[i, j]],
                    epsilon = 1e-10
                );
                assert_relative_eq!(
                    result_tiled[[i, j]],
                    result_standard[[i, j]],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]];

        let result_block = block_matmul(&a.view(), &b.view(), None);
        let result_strassen = strassen_matmul(&a.view(), &b.view(), None);
        let result_tiled = tiled_matmul(&a.view(), &b.view(), None);

        assert!(result_block.is_err());
        assert!(result_strassen.is_err());
        assert!(result_tiled.is_err());
    }
}
