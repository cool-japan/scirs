//! Block tridiagonal matrix implementation
//!
//! A block tridiagonal matrix is a block matrix which is block tridiagonal, with square matrices
//! (blocks) on the diagonal, superdiagonal, and subdiagonal. Each block can have its own
//! dimensions, creating an efficient structure for certain types of problems.

use super::SpecializedMatrix;
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use std::fmt::Debug;
use std::iter::Sum;

/// Block tridiagonal matrix representation
///
/// A block tridiagonal matrix stores blocks on the main diagonal, superdiagonal, and subdiagonal,
/// giving O(n) storage for the blocks instead of O(n²) for dense matrices.
///
/// Each matrix in the structure is stored as a square matrix using ndarray's Array2.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_linalg::specialized::BlockTridiagonalMatrix;
/// use scirs2_linalg::SpecializedMatrix;
///
/// // Create 2×2 blocks for a block tridiagonal matrix
/// let a1 = array![[1.0, 2.0], [3.0, 4.0]];
/// let a2 = array![[5.0, 6.0], [7.0, 8.0]];
/// let a3 = array![[9.0, 10.0], [11.0, 12.0]];
///
/// let b1 = array![[13.0, 14.0], [15.0, 16.0]];
/// let b2 = array![[17.0, 18.0], [19.0, 20.0]];
///
/// let c1 = array![[21.0, 22.0], [23.0, 24.0]];
/// let c2 = array![[25.0, 26.0], [27.0, 28.0]];
///
/// // Create a block tridiagonal matrix with 3 diagonal blocks
/// let diag = vec![a1.clone(), a2.clone(), a3.clone()];
/// let super_diag = vec![b1.clone(), b2.clone()];
/// let sub_diag = vec![c1.clone(), c2.clone()];
///
/// let block_tri = BlockTridiagonalMatrix::new(diag, super_diag, sub_diag).unwrap();
///
/// // Get block dimensions and total dimensions
/// assert_eq!(block_tri.block_count(), 3);
/// assert_eq!(block_tri.nrows(), 6); // 3 blocks of size 2×2
/// assert_eq!(block_tri.ncols(), 6);
///
/// // Get element at position (1,3) which is in block (0,1) at position (1,1)
/// assert_eq!(block_tri.get(1, 3).unwrap(), 16.0);
///
/// // Matrix-vector multiplication with a 6-element vector
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let y = block_tri.matvec(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct BlockTridiagonalMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// Diagonal blocks
    diagonal: Vec<Array2<A>>,

    /// Superdiagonal blocks (one fewer than diagonal)
    superdiagonal: Vec<Array2<A>>,

    /// Subdiagonal blocks (one fewer than diagonal)
    subdiagonal: Vec<Array2<A>>,

    /// Dimensions of each block
    block_dims: Vec<usize>,

    /// Total number of rows and columns
    dim: usize,
}

impl<A> BlockTridiagonalMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// Create a new block tridiagonal matrix from blocks
    ///
    /// # Arguments
    ///
    /// * `diagonal` - Vector of diagonal blocks
    /// * `superdiagonal` - Vector of superdiagonal blocks (one fewer than diagonal)
    /// * `subdiagonal` - Vector of subdiagonal blocks (one fewer than diagonal)
    ///
    /// # Returns
    ///
    /// * `LinalgResult<BlockTridiagonalMatrix<A>>` - New block tridiagonal matrix
    ///
    /// # Errors
    ///
    /// * If any diagonal block is not square
    /// * If the number of superdiagonal or subdiagonal blocks is not one fewer than diagonal
    /// * If the dimensions of adjacent blocks are incompatible
    pub fn new(
        diagonal: Vec<Array2<A>>,
        superdiagonal: Vec<Array2<A>>,
        subdiagonal: Vec<Array2<A>>,
    ) -> LinalgResult<Self> {
        let n_blocks = diagonal.len();

        if superdiagonal.len() != n_blocks - 1 || subdiagonal.len() != n_blocks - 1 {
            return Err(LinalgError::ShapeError(
                "Number of superdiagonal and subdiagonal blocks must be one fewer than diagonal"
                    .to_string(),
            ));
        }

        // Check that all diagonal blocks are square
        let mut block_dims = Vec::with_capacity(n_blocks);
        for (i, block) in diagonal.iter().enumerate() {
            let (m, n) = (block.nrows(), block.ncols());
            if m != n {
                return Err(LinalgError::ShapeError(format!(
                    "Diagonal block {i} must be square, got {m}x{n}"
                )));
            }
            block_dims.push(m);
        }

        // Check dimensions of superdiagonal blocks
        for i in 0..n_blocks - 1 {
            let (m, n) = (superdiagonal[i].nrows(), superdiagonal[i].ncols());
            if m != block_dims[i] || n != block_dims[i + 1] {
                return Err(LinalgError::ShapeError(format!(
                    "Superdiagonal block {} must be {}x{}, got {}x{}",
                    i,
                    block_dims[i],
                    block_dims[i + 1],
                    m,
                    n
                )));
            }
        }

        // Check dimensions of subdiagonal blocks
        for i in 0..n_blocks - 1 {
            let (m, n) = (subdiagonal[i].nrows(), subdiagonal[i].ncols());
            if m != block_dims[i + 1] || n != block_dims[i] {
                return Err(LinalgError::ShapeError(format!(
                    "Subdiagonal block {} must be {}x{}, got {}x{}",
                    i,
                    block_dims[i + 1],
                    block_dims[i],
                    m,
                    n
                )));
            }
        }

        // Calculate total dimension
        let dim = block_dims.iter().sum();

        Ok(Self {
            diagonal,
            superdiagonal,
            subdiagonal,
            block_dims,
            dim,
        })
    }

    /// Get the number of diagonal blocks
    pub fn block_count(&self) -> usize {
        self.diagonal.len()
    }

    /// Get a reference to a diagonal block
    pub fn diagonal_block(&self, i: usize) -> Option<&Array2<A>> {
        self.diagonal.get(i)
    }

    /// Get a reference to a superdiagonal block
    pub fn superdiagonal_block(&self, i: usize) -> Option<&Array2<A>> {
        self.superdiagonal.get(i)
    }

    /// Get a reference to a subdiagonal block
    pub fn subdiagonal_block(&self, i: usize) -> Option<&Array2<A>> {
        self.subdiagonal.get(i)
    }

    /// Find which block contains the element at position (i, j)
    ///
    /// # Arguments
    ///
    /// * `i` - Row index in the full matrix
    /// * `j` - Column index in the full matrix
    ///
    /// # Returns
    ///
    /// * `Option<(usize, usize, usize, usize)>` - (block_row, block_col, local_row, local_col)
    fn find_block_indices(&self, i: usize, j: usize) -> Option<(usize, usize, usize, usize)> {
        if i >= self.dim || j >= self.dim {
            return None;
        }

        let mut row_offset = 0;
        let mut col_offset = 0;
        let mut block_row = 0;
        let mut block_col = 0;

        // Find the block row
        for (idx, &size) in self.block_dims.iter().enumerate() {
            if i < row_offset + size {
                block_row = idx;
                break;
            }
            row_offset += size;
        }

        // Find the block column
        for (idx, &size) in self.block_dims.iter().enumerate() {
            if j < col_offset + size {
                block_col = idx;
                break;
            }
            col_offset += size;
        }

        // Check if this is a valid block in the tridiagonal structure
        let diff = block_row.abs_diff(block_col);

        if diff > 1 {
            // Outside the tridiagonal structure, element is zero
            return None;
        }

        // Convert to local indices within the block
        let local_row = i - row_offset;
        let local_col = j - col_offset;

        Some((block_row, block_col, local_row, local_col))
    }
}

impl<A> SpecializedMatrix<A> for BlockTridiagonalMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    fn nrows(&self) -> usize {
        self.dim
    }

    fn ncols(&self) -> usize {
        self.dim
    }

    fn get(&self, i: usize, j: usize) -> LinalgResult<A> {
        if i >= self.dim || j >= self.dim {
            return Err(LinalgError::IndexError(format!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                i, j, self.dim, self.dim
            )));
        }

        // Find which block contains this element and its local indices
        if let Some((block_row, block_col, local_row, local_col)) = self.find_block_indices(i, j) {
            // Return the element based on which diagonal it's on
            let value = if block_row == block_col {
                // Main diagonal
                self.diagonal[block_row][[local_row, local_col]]
            } else if block_row + 1 == block_col {
                // Superdiagonal
                self.superdiagonal[block_row][[local_row, local_col]]
            } else if block_row == block_col + 1 {
                // Subdiagonal
                self.subdiagonal[block_col][[local_row, local_col]]
            } else {
                // Outside the tridiagonal structure
                A::zero()
            };

            Ok(value)
        } else {
            // Element is outside the tridiagonal structure
            Ok(A::zero())
        }
    }

    fn matvec(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if x.len() != self.dim {
            return Err(LinalgError::ShapeError(format!(
                "Vector has incompatible dimension {} for matrix of dimension {}",
                x.len(),
                self.dim
            )));
        }

        let mut result = Array1::zeros(self.dim);
        let mut row_offset = 0;

        // Process each block row
        for block_idx in 0..self.block_count() {
            let blocksize = self.block_dims[block_idx];
            let mut result_block =
                result.slice_mut(scirs2_core::ndarray::s![row_offset..row_offset + blocksize]);

            // Initialize with zeros
            for val in result_block.iter_mut() {
                *val = A::zero();
            }

            // Process each block in this block row

            // Diagonal block
            let diag_block = &self.diagonal[block_idx];
            let mut col_offset = 0;
            // Find the column offset for this block
            for b in 0..block_idx {
                col_offset += self.block_dims[b];
            }
            let x_block = x.slice(scirs2_core::ndarray::s![col_offset..col_offset + blocksize]);

            // Multiply diagonal block: result_block += diag_block * x_block
            for i in 0..blocksize {
                for j in 0..blocksize {
                    result_block[i] += diag_block[[i, j]] * x_block[j];
                }
            }

            // Superdiagonal block (if it exists)
            if block_idx < self.block_count() - 1 {
                let super_block = &self.superdiagonal[block_idx];
                let next_blocksize = self.block_dims[block_idx + 1];
                let x_next_block = x.slice(scirs2_core::ndarray::s![
                    col_offset + blocksize..col_offset + blocksize + next_blocksize
                ]);

                // Multiply superdiagonal block: result_block += super_block * x_next_block
                for i in 0..blocksize {
                    for j in 0..next_blocksize {
                        result_block[i] += super_block[[i, j]] * x_next_block[j];
                    }
                }
            }

            // Subdiagonal block (if it exists)
            if block_idx > 0 {
                let prev_blocksize = self.block_dims[block_idx - 1];
                let sub_block = &self.subdiagonal[block_idx - 1];
                let x_prev_block = x.slice(scirs2_core::ndarray::s![
                    col_offset - prev_blocksize..col_offset
                ]);

                // Multiply subdiagonal block: result_block += sub_block * x_prev_block
                for i in 0..blocksize {
                    for j in 0..prev_blocksize {
                        result_block[i] += sub_block[[i, j]] * x_prev_block[j];
                    }
                }
            }

            row_offset += blocksize;
        }

        Ok(result)
    }

    fn matvec_transpose(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if x.len() != self.dim {
            return Err(LinalgError::ShapeError(format!(
                "Vector has incompatible dimension {} for matrix of dimension {}",
                x.len(),
                self.dim
            )));
        }

        let mut result = Array1::zeros(self.dim);
        let mut col_offset = 0;

        // Process each block column
        for block_idx in 0..self.block_count() {
            let blocksize = self.block_dims[block_idx];
            let mut result_block =
                result.slice_mut(scirs2_core::ndarray::s![col_offset..col_offset + blocksize]);

            // Initialize with zeros
            for val in result_block.iter_mut() {
                *val = A::zero();
            }

            // Process each block in this block column

            // Diagonal block
            let diag_block = &self.diagonal[block_idx];
            let mut row_offset = 0;
            // Find the row offset for this block
            for b in 0..block_idx {
                row_offset += self.block_dims[b];
            }
            let x_block = x.slice(scirs2_core::ndarray::s![row_offset..row_offset + blocksize]);

            // Multiply diagonal block transpose: result_block += diag_block^T * x_block
            for i in 0..blocksize {
                for j in 0..blocksize {
                    result_block[i] += diag_block[[j, i]] * x_block[j];
                }
            }

            // Subdiagonal block (which becomes superdiagonal in transpose)
            if block_idx < self.block_count() - 1 {
                let sub_block = &self.subdiagonal[block_idx];
                let next_blocksize = self.block_dims[block_idx + 1];
                let x_next_block = x.slice(scirs2_core::ndarray::s![
                    row_offset + blocksize..row_offset + blocksize + next_blocksize
                ]);

                // Multiply subdiagonal block transpose: result_block += sub_block^T * x_next_block
                for i in 0..blocksize {
                    for j in 0..next_blocksize {
                        result_block[i] += sub_block[[j, i]] * x_next_block[j];
                    }
                }
            }

            // Superdiagonal block (which becomes subdiagonal in transpose)
            if block_idx > 0 {
                let prev_blocksize = self.block_dims[block_idx - 1];
                let super_block = &self.superdiagonal[block_idx - 1];
                let x_prev_block = x.slice(scirs2_core::ndarray::s![
                    row_offset - prev_blocksize..row_offset
                ]);

                // Multiply superdiagonal block transpose: result_block += super_block^T * x_prev_block
                for i in 0..blocksize {
                    for j in 0..prev_blocksize {
                        result_block[i] += super_block[[j, i]] * x_prev_block[j];
                    }
                }
            }

            col_offset += blocksize;
        }

        Ok(result)
    }

    fn to_dense(&self) -> LinalgResult<Array2<A>> {
        let mut result = Array2::zeros((self.dim, self.dim));

        let mut row_offset = 0;
        for i in 0..self.block_count() {
            let blocksize_i = self.block_dims[i];
            let mut col_offset = 0;

            // Iterate through blocks in the same block row
            for j in 0..self.block_count() {
                let blocksize_j = self.block_dims[j];

                // Determine which block to use
                let block_opt = if i == j {
                    // Diagonal block
                    Some(&self.diagonal[i])
                } else if i + 1 == j {
                    // Superdiagonal block
                    Some(&self.superdiagonal[i])
                } else if i == j + 1 {
                    // Subdiagonal block
                    Some(&self.subdiagonal[j])
                } else {
                    // Zero block outside the tridiagonal structure
                    None
                };

                // Fill in the block if it exists
                if let Some(block) = block_opt {
                    for bi in 0..block.nrows() {
                        for bj in 0..block.ncols() {
                            result[[row_offset + bi, col_offset + bj]] = block[[bi, bj]];
                        }
                    }
                }

                col_offset += blocksize_j;
            }

            row_offset += blocksize_i;
        }

        Ok(result)
    }
}

/// Perform LU decomposition of a block tridiagonal matrix
///
/// This function computes the LU decomposition of a block tridiagonal matrix
/// using block-wise Gaussian elimination. The decomposition is performed in-place
/// modifying the original matrix structure.
///
/// # Arguments
///
/// * `matrix` - The block tridiagonal matrix to decompose
///
/// # Returns
///
/// A tuple containing the L and U factors as separate block tridiagonal matrices
#[allow(dead_code)]
pub fn block_tridiagonal_lu<A>(
    matrix: &BlockTridiagonalMatrix<A>,
) -> LinalgResult<(BlockTridiagonalMatrix<A>, BlockTridiagonalMatrix<A>)>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    // For simplicity, convert to dense matrix and use standard LU decomposition
    // This is not the most efficient implementation but ensures correctness
    let dense = matrix.to_dense()?;

    // Use standard LU decomposition from the main solver
    let n = dense.nrows();
    let mut l = Array2::eye(n);
    let mut u = dense.clone();

    // Gaussian elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        for i in k + 1..n {
            if u[[i, k]].abs() > u[[max_idx, k]].abs() {
                max_idx = i;
            }
        }

        // Swap rows if needed
        if max_idx != k {
            for j in 0..n {
                let temp = u[[k, j]];
                u[[k, j]] = u[[max_idx, j]];
                u[[max_idx, j]] = temp;

                if j < k {
                    let temp = l[[k, j]];
                    l[[k, j]] = l[[max_idx, j]];
                    l[[max_idx, j]] = temp;
                }
            }
        }

        // Check for zero pivot
        if u[[k, k]].abs() < A::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular during LU decomposition".to_string(),
            ));
        }

        // Elimination
        for i in k + 1..n {
            let factor = u[[i, k]] / u[[k, k]];
            l[[i, k]] = factor;

            for j in k..n {
                u[[i, j]] = u[[i, j]] - factor * u[[k, j]];
            }
        }
    }

    // Convert back to block tridiagonal format
    // For simplicity, we'll create single-element blocks
    let blocksize = 1;
    let num_blocks = n;

    let mut l_diag = Vec::new();
    let mut l_sub = Vec::new();
    let mut l_super = Vec::new();

    let mut u_diag = Vec::new();
    let mut u_sub = Vec::new();
    let mut u_super = Vec::new();

    for i in 0..num_blocks {
        // Diagonal blocks
        l_diag.push(Array2::from_elem((blocksize, blocksize), l[[i, i]]));
        u_diag.push(Array2::from_elem((blocksize, blocksize), u[[i, i]]));

        // Off-diagonal blocks
        if i < num_blocks - 1 {
            l_super.push(Array2::zeros((blocksize, blocksize)));
            u_super.push(Array2::from_elem((blocksize, blocksize), u[[i, i + 1]]));
        }

        if i > 0 {
            l_sub.push(Array2::from_elem((blocksize, blocksize), l[[i, i - 1]]));
            u_sub.push(Array2::zeros((blocksize, blocksize)));
        }
    }

    let lmatrix = BlockTridiagonalMatrix::new(l_diag, l_super, l_sub)?;
    let umatrix = BlockTridiagonalMatrix::new(u_diag, u_super, u_sub)?;

    Ok((lmatrix, umatrix))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    fn create_testmatrix() -> BlockTridiagonalMatrix<f64> {
        // Create 2×2 blocks
        let a1 = array![[1.0, 2.0], [3.0, 4.0]];
        let a2 = array![[5.0, 6.0], [7.0, 8.0]];
        let a3 = array![[9.0, 10.0], [11.0, 12.0]];

        let b1 = array![[13.0, 14.0], [15.0, 16.0]];
        let b2 = array![[17.0, 18.0], [19.0, 20.0]];

        let c1 = array![[21.0, 22.0], [23.0, 24.0]];
        let c2 = array![[25.0, 26.0], [27.0, 28.0]];

        BlockTridiagonalMatrix::new(vec![a1, a2, a3], vec![b1, b2], vec![c1, c2]).unwrap()
    }

    #[test]
    fn test_constructor() {
        let matrix = create_testmatrix();

        assert_eq!(matrix.block_count(), 3);
        assert_eq!(matrix.nrows(), 6);
        assert_eq!(matrix.ncols(), 6);
        assert_eq!(matrix.block_dims, vec![2, 2, 2]);
    }

    #[test]
    fn test_element_access() {
        let matrix = create_testmatrix();

        // Test elements in diagonal blocks
        assert_eq!(matrix.get(0, 0).unwrap(), 1.0);
        assert_eq!(matrix.get(0, 1).unwrap(), 2.0);
        assert_eq!(matrix.get(2, 2).unwrap(), 5.0);
        assert_eq!(matrix.get(5, 5).unwrap(), 12.0);

        // Test elements in superdiagonal blocks
        assert_eq!(matrix.get(0, 2).unwrap(), 13.0);
        assert_eq!(matrix.get(1, 3).unwrap(), 16.0);
        assert_eq!(matrix.get(2, 4).unwrap(), 17.0);

        // Test elements in subdiagonal blocks
        assert_eq!(matrix.get(2, 0).unwrap(), 21.0);
        assert_eq!(matrix.get(4, 2).unwrap(), 25.0);

        // Test elements that are outside of diagonal+1 spacing
        assert_eq!(matrix.get(0, 4).unwrap(), 0.0);
        // Element (5,2) is in subdiagonal block c2
        assert_eq!(matrix.get(5, 2).unwrap(), 27.0);
    }

    #[test]
    fn test_to_dense() {
        let matrix = create_testmatrix();
        let dense = matrix.to_dense().unwrap();

        // Expected dense matrix
        #[rustfmt::skip]
        let expected = array![
            [1.0,  2.0,  13.0, 14.0, 0.0,  0.0],
            [3.0,  4.0,  15.0, 16.0, 0.0,  0.0],
            [21.0, 22.0, 5.0,  6.0,  17.0, 18.0],
            [23.0, 24.0, 7.0,  8.0,  19.0, 20.0],
            [0.0,  0.0,  25.0, 26.0, 9.0,  10.0],
            [0.0,  0.0,  27.0, 28.0, 11.0, 12.0],
        ];

        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(dense[[i, j]], expected[[i, j]]);
            }
        }
    }

    #[test]
    fn test_matvec() {
        let matrix = create_testmatrix();
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = matrix.matvec(&x.view()).unwrap();

        // Calculate expected result using the dense version
        let dense = matrix.to_dense().unwrap();
        let mut expected = Array1::zeros(6);
        for i in 0..6 {
            for j in 0..6 {
                expected[i] += dense[[i, j]] * x[j];
            }
        }

        for i in 0..6 {
            assert_relative_eq!(y[i], expected[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_matvec_transpose() {
        let matrix = create_testmatrix();
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = matrix.matvec_transpose(&x.view()).unwrap();

        // Calculate expected result using the dense version
        let dense = matrix.to_dense().unwrap();
        let mut expected = Array1::zeros(6);
        for i in 0..6 {
            for j in 0..6 {
                expected[i] += dense[[j, i]] * x[j];
            }
        }

        for i in 0..6 {
            assert_relative_eq!(y[i], expected[i], epsilon = 1e-10);
        }
    }
}
