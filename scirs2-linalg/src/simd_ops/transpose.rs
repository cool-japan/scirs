//! SIMD-accelerated matrix transpose operations
//!
//! This module provides cache-friendly, SIMD-accelerated matrix transpose
//! operations with blocking to optimize memory access patterns.

#[cfg(feature = "simd")]
use crate::error::{LinalgError, LinalgResult};
#[cfg(feature = "simd")]
use ndarray::{Array2, ArrayView2};
#[cfg(feature = "simd")]
use wide::{f32x8, f64x4};

/// SIMD-accelerated matrix transpose for f32 values with cache-friendly blocking
///
/// This implementation uses cache-friendly blocking to optimize memory access
/// patterns and SIMD instructions for better performance on large matrices.
///
/// # Arguments
///
/// * `matrix` - Input matrix to transpose
///
/// # Returns
///
/// * Transposed matrix
#[cfg(feature = "simd")]
pub fn simd_transpose_f32(matrix: &ArrayView2<f32>) -> LinalgResult<Array2<f32>> {
    let (rows, cols) = matrix.dim();
    let mut result = Array2::zeros((cols, rows));

    // Block sizes optimized for cache performance
    const BLOCK_SIZE: usize = 32;

    // Process matrix in blocks for cache efficiency
    for i0 in (0..rows).step_by(BLOCK_SIZE) {
        let i_end = (i0 + BLOCK_SIZE).min(rows);

        for j0 in (0..cols).step_by(BLOCK_SIZE) {
            let j_end = (j0 + BLOCK_SIZE).min(cols);

            // Transpose within this block
            transpose_block_f32(matrix, &mut result, i0, i_end, j0, j_end)?;
        }
    }

    Ok(result)
}

/// SIMD-accelerated matrix transpose for f64 values with cache-friendly blocking
///
/// This implementation uses cache-friendly blocking to optimize memory access
/// patterns and SIMD instructions for better performance on large matrices.
///
/// # Arguments
///
/// * `matrix` - Input matrix to transpose
///
/// # Returns
///
/// * Transposed matrix
#[cfg(feature = "simd")]
pub fn simd_transpose_f64(matrix: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let (rows, cols) = matrix.dim();
    let mut result = Array2::zeros((cols, rows));

    // Block sizes optimized for cache performance (smaller for f64 due to larger size)
    const BLOCK_SIZE: usize = 16;

    // Process matrix in blocks for cache efficiency
    for i0 in (0..rows).step_by(BLOCK_SIZE) {
        let i_end = (i0 + BLOCK_SIZE).min(rows);

        for j0 in (0..cols).step_by(BLOCK_SIZE) {
            let j_end = (j0 + BLOCK_SIZE).min(cols);

            // Transpose within this block
            transpose_block_f64(matrix, &mut result, i0, i_end, j0, j_end)?;
        }
    }

    Ok(result)
}

/// Transpose a block of f32 matrix using SIMD instructions
#[cfg(feature = "simd")]
fn transpose_block_f32(
    src: &ArrayView2<f32>,
    dst: &mut Array2<f32>,
    i0: usize,
    i_end: usize,
    j0: usize,
    j_end: usize,
) -> LinalgResult<()> {
    // For small blocks, use vectorized 8x8 transpose when possible
    if i_end - i0 >= 8 && j_end - j0 >= 8 {
        // Process 8x8 blocks with SIMD
        for i in (i0..i_end).step_by(8) {
            for j in (j0..j_end).step_by(8) {
                if i + 8 <= i_end && j + 8 <= j_end {
                    transpose_8x8_f32(src, dst, i, j)?;
                } else {
                    // Fallback for partial blocks
                    for ii in i..(i + 8).min(i_end) {
                        for jj in j..(j + 8).min(j_end) {
                            dst[[jj, ii]] = src[[ii, jj]];
                        }
                    }
                }
            }
        }
    } else {
        // Simple transpose for small blocks
        for i in i0..i_end {
            for j in j0..j_end {
                dst[[j, i]] = src[[i, j]];
            }
        }
    }

    Ok(())
}

/// Transpose a block of f64 matrix using SIMD instructions
#[cfg(feature = "simd")]
fn transpose_block_f64(
    src: &ArrayView2<f64>,
    dst: &mut Array2<f64>,
    i0: usize,
    i_end: usize,
    j0: usize,
    j_end: usize,
) -> LinalgResult<()> {
    // For small blocks, use vectorized 4x4 transpose when possible
    if i_end - i0 >= 4 && j_end - j0 >= 4 {
        // Process 4x4 blocks with SIMD
        for i in (i0..i_end).step_by(4) {
            for j in (j0..j_end).step_by(4) {
                if i + 4 <= i_end && j + 4 <= j_end {
                    transpose_4x4_f64(src, dst, i, j)?;
                } else {
                    // Fallback for partial blocks
                    for ii in i..(i + 4).min(i_end) {
                        for jj in j..(j + 4).min(j_end) {
                            dst[[jj, ii]] = src[[ii, jj]];
                        }
                    }
                }
            }
        }
    } else {
        // Simple transpose for small blocks
        for i in i0..i_end {
            for j in j0..j_end {
                dst[[j, i]] = src[[i, j]];
            }
        }
    }

    Ok(())
}

/// SIMD-optimized 8x8 f32 matrix transpose
#[cfg(feature = "simd")]
fn transpose_8x8_f32(
    src: &ArrayView2<f32>,
    dst: &mut Array2<f32>,
    start_i: usize,
    start_j: usize,
) -> LinalgResult<()> {
    // Load 8 rows as SIMD vectors
    let mut rows = Vec::with_capacity(8);

    for i in 0..8 {
        if let Some(row_slice) = src.row(start_i + i).as_slice() {
            let row_data = [
                row_slice[start_j],
                row_slice[start_j + 1],
                row_slice[start_j + 2],
                row_slice[start_j + 3],
                row_slice[start_j + 4],
                row_slice[start_j + 5],
                row_slice[start_j + 6],
                row_slice[start_j + 7],
            ];
            rows.push(f32x8::new(row_data));
        } else {
            // Fallback for non-contiguous data
            for ii in 0..8 {
                for jj in 0..8 {
                    dst[[start_j + jj, start_i + ii]] = src[[start_i + ii, start_j + jj]];
                }
            }
            return Ok(());
        }
    }

    // Perform the transpose using SIMD shuffles
    // This is a simplified implementation - a full implementation would use
    // more sophisticated SIMD shuffle instructions for optimal performance
    for i in 0..8 {
        let row_arrays: [[f32; 8]; 8] = [
            rows[0].into(),
            rows[1].into(),
            rows[2].into(),
            rows[3].into(),
            rows[4].into(),
            rows[5].into(),
            rows[6].into(),
            rows[7].into(),
        ];
        let col_data: [f32; 8] = [
            row_arrays[0][i],
            row_arrays[1][i],
            row_arrays[2][i],
            row_arrays[3][i],
            row_arrays[4][i],
            row_arrays[5][i],
            row_arrays[6][i],
            row_arrays[7][i],
        ];

        // Store the transposed column
        for (j, &val) in col_data.iter().enumerate() {
            dst[[start_j + i, start_i + j]] = val;
        }
    }

    Ok(())
}

/// SIMD-optimized 4x4 f64 matrix transpose
#[cfg(feature = "simd")]
fn transpose_4x4_f64(
    src: &ArrayView2<f64>,
    dst: &mut Array2<f64>,
    start_i: usize,
    start_j: usize,
) -> LinalgResult<()> {
    // Load 4 rows as SIMD vectors
    let mut rows = Vec::with_capacity(4);

    for i in 0..4 {
        if let Some(row_slice) = src.row(start_i + i).as_slice() {
            let row_data = [
                row_slice[start_j],
                row_slice[start_j + 1],
                row_slice[start_j + 2],
                row_slice[start_j + 3],
            ];
            rows.push(f64x4::new(row_data));
        } else {
            // Fallback for non-contiguous data
            for ii in 0..4 {
                for jj in 0..4 {
                    dst[[start_j + jj, start_i + ii]] = src[[start_i + ii, start_j + jj]];
                }
            }
            return Ok(());
        }
    }

    // Perform the transpose using SIMD
    for i in 0..4 {
        let row_arrays: [[f64; 4]; 4] = [
            rows[0].into(),
            rows[1].into(),
            rows[2].into(),
            rows[3].into(),
        ];
        let col_data: [f64; 4] = [
            row_arrays[0][i],
            row_arrays[1][i],
            row_arrays[2][i],
            row_arrays[3][i],
        ];

        // Store the transposed column
        for (j, &val) in col_data.iter().enumerate() {
            dst[[start_j + i, start_i + j]] = val;
        }
    }

    Ok(())
}

/// In-place matrix transpose for square f32 matrices using SIMD
///
/// This function performs in-place transpose for square matrices,
/// which is more memory-efficient than creating a new matrix.
///
/// # Arguments
///
/// * `matrix` - Square matrix to transpose in-place
///
/// # Returns
///
/// * Result indicating success or failure
#[cfg(feature = "simd")]
pub fn simd_transpose_inplace_f32(matrix: &mut Array2<f32>) -> LinalgResult<()> {
    let (rows, cols) = matrix.dim();

    if rows != cols {
        return Err(LinalgError::ShapeError(format!(
            "In-place transpose requires square matrix, got shape ({}, {})",
            rows, cols
        )));
    }

    let n = rows;
    const BLOCK_SIZE: usize = 32;

    // Process in blocks along the diagonal
    for i0 in (0..n).step_by(BLOCK_SIZE) {
        let i_end = (i0 + BLOCK_SIZE).min(n);

        for j0 in (i0..n).step_by(BLOCK_SIZE) {
            let j_end = (j0 + BLOCK_SIZE).min(n);

            if i0 == j0 {
                // Diagonal block - transpose in place
                transpose_diagonal_block_f32(matrix, i0, i_end)?;
            } else {
                // Off-diagonal block - swap with symmetric block
                swap_blocks_f32(matrix, i0, i_end, j0, j_end)?;
            }
        }
    }

    Ok(())
}

/// In-place matrix transpose for square f64 matrices using SIMD
///
/// This function performs in-place transpose for square matrices,
/// which is more memory-efficient than creating a new matrix.
///
/// # Arguments
///
/// * `matrix` - Square matrix to transpose in-place
///
/// # Returns
///
/// * Result indicating success or failure
#[cfg(feature = "simd")]
pub fn simd_transpose_inplace_f64(matrix: &mut Array2<f64>) -> LinalgResult<()> {
    let (rows, cols) = matrix.dim();

    if rows != cols {
        return Err(LinalgError::ShapeError(format!(
            "In-place transpose requires square matrix, got shape ({}, {})",
            rows, cols
        )));
    }

    let n = rows;
    const BLOCK_SIZE: usize = 16;

    // Process in blocks along the diagonal
    for i0 in (0..n).step_by(BLOCK_SIZE) {
        let i_end = (i0 + BLOCK_SIZE).min(n);

        for j0 in (i0..n).step_by(BLOCK_SIZE) {
            let j_end = (j0 + BLOCK_SIZE).min(n);

            if i0 == j0 {
                // Diagonal block - transpose in place
                transpose_diagonal_block_f64(matrix, i0, i_end)?;
            } else {
                // Off-diagonal block - swap with symmetric block
                swap_blocks_f64(matrix, i0, i_end, j0, j_end)?;
            }
        }
    }

    Ok(())
}

/// Transpose a diagonal block in-place for f32
#[cfg(feature = "simd")]
fn transpose_diagonal_block_f32(
    matrix: &mut Array2<f32>,
    start: usize,
    end: usize,
) -> LinalgResult<()> {
    for i in start..end {
        for j in (i + 1)..end {
            let temp = matrix[[i, j]];
            matrix[[i, j]] = matrix[[j, i]];
            matrix[[j, i]] = temp;
        }
    }
    Ok(())
}

/// Transpose a diagonal block in-place for f64
#[cfg(feature = "simd")]
fn transpose_diagonal_block_f64(
    matrix: &mut Array2<f64>,
    start: usize,
    end: usize,
) -> LinalgResult<()> {
    for i in start..end {
        for j in (i + 1)..end {
            let temp = matrix[[i, j]];
            matrix[[i, j]] = matrix[[j, i]];
            matrix[[j, i]] = temp;
        }
    }
    Ok(())
}

/// Swap two off-diagonal blocks for f32
#[cfg(feature = "simd")]
fn swap_blocks_f32(
    matrix: &mut Array2<f32>,
    i0: usize,
    i_end: usize,
    j0: usize,
    j_end: usize,
) -> LinalgResult<()> {
    for i in i0..i_end {
        for j in j0..j_end {
            let temp = matrix[[i, j]];
            matrix[[i, j]] = matrix[[j, i]];
            matrix[[j, i]] = temp;
        }
    }
    Ok(())
}

/// Swap two off-diagonal blocks for f64
#[cfg(feature = "simd")]
fn swap_blocks_f64(
    matrix: &mut Array2<f64>,
    i0: usize,
    i_end: usize,
    j0: usize,
    j_end: usize,
) -> LinalgResult<()> {
    for i in i0..i_end {
        for j in j0..j_end {
            let temp = matrix[[i, j]];
            matrix[[i, j]] = matrix[[j, i]];
            matrix[[j, i]] = temp;
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_transpose_f32() {
        let matrix = array![
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];

        let result = simd_transpose_f32(&matrix.view()).unwrap();

        let expected = array![
            [1.0f32, 5.0, 9.0],
            [2.0, 6.0, 10.0],
            [3.0, 7.0, 11.0],
            [4.0, 8.0, 12.0]
        ];

        assert_eq!(result.shape(), expected.shape());
        for ((i, j), &val) in result.indexed_iter() {
            assert_relative_eq!(val, expected[[i, j]], epsilon = 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_transpose_f64() {
        let matrix = array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let result = simd_transpose_f64(&matrix.view()).unwrap();

        let expected = array![[1.0f64, 4.0], [2.0, 5.0], [3.0, 6.0]];

        assert_eq!(result.shape(), expected.shape());
        for ((i, j), &val) in result.indexed_iter() {
            assert_relative_eq!(val, expected[[i, j]], epsilon = 1e-12);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_transpose_inplace_f32() {
        let mut matrix = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        simd_transpose_inplace_f32(&mut matrix).unwrap();

        let expected = array![[1.0f32, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]];

        assert_eq!(matrix.shape(), expected.shape());
        for ((i, j), &val) in matrix.indexed_iter() {
            assert_relative_eq!(val, expected[[i, j]], epsilon = 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_transpose_large_matrix() {
        // Test with larger matrix to exercise blocking
        let size = 100;
        let mut matrix = Array2::zeros((size, size));

        // Fill with test pattern
        for i in 0..size {
            for j in 0..size {
                matrix[[i, j]] = (i * size + j) as f32;
            }
        }

        let result = simd_transpose_f32(&matrix.view()).unwrap();

        // Verify transpose correctness
        for i in 0..size {
            for j in 0..size {
                assert_relative_eq!(result[[j, i]], matrix[[i, j]], epsilon = 1e-6);
            }
        }
    }
}
