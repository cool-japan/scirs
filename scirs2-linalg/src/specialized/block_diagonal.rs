//! Block diagonal matrix implementation
//!
//! This module provides specialized storage and algorithms for block diagonal matrices,
//! which consist of square blocks along the main diagonal with zeros elsewhere.
//! Such matrices arise naturally in many applications and allow for highly efficient
//! parallel and blocked operations.

use super::SpecializedMatrix;
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use std::fmt::Debug;
use std::iter::Sum;

/// Block diagonal matrix with efficient storage and operations
///
/// Stores only the blocks along the diagonal, enabling O(sum of block sizes) storage
/// instead of O(n²) for the full matrix.
#[derive(Debug, Clone)]
pub struct BlockDiagonalMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// Diagonal blocks
    blocks: Vec<Array2<A>>,
    /// Total size of the matrix
    size: usize,
    /// Starting row/column index for each block
    block_offsets: Vec<usize>,
}

impl<A> BlockDiagonalMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    /// Create a new block diagonal matrix from a list of square blocks
    pub fn new(blocks: Vec<Array2<A>>) -> LinalgResult<Self> {
        if blocks.is_empty() {
            return Err(LinalgError::InvalidInput(
                "At least one block is required".to_string(),
            ));
        }

        // Check that all _blocks are square
        for (i, block) in blocks.iter().enumerate() {
            if block.nrows() != block.ncols() {
                return Err(LinalgError::InvalidInput(format!(
                    "Block {} is not square: {}x{}",
                    i,
                    block.nrows(),
                    block.ncols()
                )));
            }
        }

        // Calculate block offsets and total size
        let mut block_offsets = Vec::with_capacity(blocks.len());
        let mut offset = 0;
        for block in &blocks {
            block_offsets.push(offset);
            offset += block.nrows();
        }
        let size = offset;

        Ok(Self {
            blocks,
            size,
            block_offsets,
        })
    }

    /// Create a block diagonal matrix from a list of array views
    pub fn from_views(blocks: Vec<ArrayView2<A>>) -> LinalgResult<Self> {
        let owned_blocks: Vec<Array2<A>> = blocks.into_iter().map(|b| b.to_owned()).collect();
        Self::new(owned_blocks)
    }

    /// Get the number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get a reference to a specific block
    pub fn block(&self, index: usize) -> Option<&Array2<A>> {
        self.blocks.get(index)
    }

    /// Get a mutable reference to a specific block
    pub fn block_mut(&mut self, index: usize) -> Option<&mut Array2<A>> {
        self.blocks.get_mut(index)
    }

    /// Get the size of a specific block
    pub fn blocksize(&self, index: usize) -> Option<usize> {
        self.blocks.get(index).map(|b| b.nrows())
    }

    /// Get the starting offset for a specific block
    pub fn block_offset(&self, index: usize) -> Option<usize> {
        self.block_offsets.get(index).copied()
    }

    /// Find which block contains the given row/column index
    fn find_block(&self, index: usize) -> Option<usize> {
        if index >= self.size {
            return None;
        }

        for (block_idx, &offset) in self.block_offsets.iter().enumerate() {
            let blocksize = self.blocks[block_idx].nrows();
            if index >= offset && index < offset + blocksize {
                return Some(block_idx);
            }
        }
        None
    }

    /// Solve the linear system Ax = b using block-wise operations
    pub fn solve(&self, b: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if b.len() != self.size {
            return Err(LinalgError::ShapeError(format!(
                "Right-hand side has length {}, expected {}",
                b.len(),
                self.size
            )));
        }

        let mut x = Array1::zeros(self.size);

        // Solve each block independently
        for (block_idx, block) in self.blocks.iter().enumerate() {
            let offset = self.block_offsets[block_idx];
            let blocksize = block.nrows();

            let b_block = b.slice(scirs2_core::ndarray::s![offset..offset + blocksize]);
            let x_block = crate::solve::solve(
                &block.view(),
                &b_block,
                Some(1), // workers
            )?;

            x.slice_mut(scirs2_core::ndarray::s![offset..offset + blocksize])
                .assign(&x_block);
        }

        Ok(x)
    }

    /// Compute the determinant as the product of block determinants
    pub fn determinant(&self) -> LinalgResult<A> {
        let mut det = A::one();

        for block in &self.blocks {
            let block_det = crate::basic::det(&block.view(), Some(1))?; // workers = 1
            det *= block_det;
        }

        Ok(det)
    }

    /// Compute the inverse as a block diagonal matrix of block inverses
    pub fn inverse(&self) -> LinalgResult<Self> {
        let mut inv_blocks = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            let inv_block = crate::basic::inv(&block.view(), Some(1))?; // workers = 1
            inv_blocks.push(inv_block);
        }

        Self::new(inv_blocks)
    }

    /// Compute the trace as the sum of block traces
    pub fn trace(&self) -> A {
        let mut trace = A::zero();
        for block in &self.blocks {
            for i in 0..block.nrows() {
                trace += block[[i, i]];
            }
        }
        trace
    }
}

impl<A> SpecializedMatrix<A> for BlockDiagonalMatrix<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    fn nrows(&self) -> usize {
        self.size
    }

    fn ncols(&self) -> usize {
        self.size
    }

    fn get(&self, i: usize, j: usize) -> LinalgResult<A> {
        if i >= self.size || j >= self.size {
            return Err(LinalgError::IndexError(format!(
                "Index ({}, {}) out of bounds for {}x{} matrix",
                i, j, self.size, self.size
            )));
        }

        let block_i = self.find_block(i);
        let block_j = self.find_block(j);

        match (block_i, block_j) {
            (Some(bi), Some(bj)) if bi == bj => {
                // Element is within a diagonal block
                let offset = self.block_offsets[bi];
                let local_i = i - offset;
                let local_j = j - offset;
                Ok(self.blocks[bi][[local_i, local_j]])
            }
            _ => {
                // Element is outside diagonal blocks, so it's zero
                Ok(A::zero())
            }
        }
    }

    fn matvec(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if x.len() != self.size {
            return Err(LinalgError::ShapeError(format!(
                "Vector has length {}, expected {}",
                x.len(),
                self.size
            )));
        }

        let mut result = Array1::zeros(self.size);

        // Multiply each block independently
        for (block_idx, block) in self.blocks.iter().enumerate() {
            let offset = self.block_offsets[block_idx];
            let blocksize = block.nrows();

            let x_block = x.slice(scirs2_core::ndarray::s![offset..offset + blocksize]);
            let y_block = block.dot(&x_block);

            result
                .slice_mut(scirs2_core::ndarray::s![offset..offset + blocksize])
                .assign(&y_block);
        }

        Ok(result)
    }

    fn matvec_transpose(&self, x: &ArrayView1<A>) -> LinalgResult<Array1<A>> {
        if x.len() != self.size {
            return Err(LinalgError::ShapeError(format!(
                "Vector has length {}, expected {}",
                x.len(),
                self.size
            )));
        }

        let mut result = Array1::zeros(self.size);

        // Multiply each block transpose independently
        for (block_idx, block) in self.blocks.iter().enumerate() {
            let offset = self.block_offsets[block_idx];
            let blocksize = block.nrows();

            let x_block = x.slice(scirs2_core::ndarray::s![offset..offset + blocksize]);
            let block_t = block.t();
            let y_block = block_t.dot(&x_block);

            result
                .slice_mut(scirs2_core::ndarray::s![offset..offset + blocksize])
                .assign(&y_block);
        }

        Ok(result)
    }

    fn to_dense(&self) -> LinalgResult<Array2<A>> {
        let mut dense = Array2::zeros((self.size, self.size));

        for (block_idx, block) in self.blocks.iter().enumerate() {
            let offset = self.block_offsets[block_idx];
            let blocksize = block.nrows();

            let mut dense_block = dense.slice_mut(scirs2_core::ndarray::s![
                offset..offset + blocksize,
                offset..offset + blocksize
            ]);
            dense_block.assign(block);
        }

        Ok(dense)
    }
}

/// Solve a block diagonal linear system
#[allow(dead_code)]
pub fn solve_block_diagonal<A>(
    matrix: &BlockDiagonalMatrix<A>,
    b: &ArrayView1<A>,
) -> LinalgResult<Array1<A>>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    matrix.solve(b)
}

/// Compute the determinant of a block diagonal matrix
#[allow(dead_code)]
pub fn block_diagonal_determinant<A>(matrix: &BlockDiagonalMatrix<A>) -> LinalgResult<A>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    matrix.determinant()
}

/// Create a block diagonal matrix from a list of arrays
#[allow(dead_code)]
pub fn create_block_diagonal<A>(blocks: Vec<Array2<A>>) -> LinalgResult<BlockDiagonalMatrix<A>>
where
    A: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync + Debug,
{
    BlockDiagonalMatrix::new(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_block_diagonal_creation() {
        let block1 = array![[1.0, 2.0], [3.0, 4.0]];
        let block2 = array![[5.0, 6.0], [7.0, 8.0]];
        let block3 = array![[9.0]];

        let blocks = vec![block1, block2, block3];
        let bdmatrix = BlockDiagonalMatrix::new(blocks).unwrap();

        assert_eq!(bdmatrix.size, 5);
        assert_eq!(bdmatrix.num_blocks(), 3);
        assert_eq!(bdmatrix.blocksize(0), Some(2));
        assert_eq!(bdmatrix.blocksize(1), Some(2));
        assert_eq!(bdmatrix.blocksize(2), Some(1));
    }

    #[test]
    fn test_block_diagonal_get() {
        let block1 = array![[1.0, 2.0], [3.0, 4.0]];
        let block2 = array![[5.0]];
        let blocks = vec![block1, block2];
        let bdmatrix = BlockDiagonalMatrix::new(blocks).unwrap();

        assert_eq!(bdmatrix.get(0, 0).unwrap(), 1.0);
        assert_eq!(bdmatrix.get(0, 1).unwrap(), 2.0);
        assert_eq!(bdmatrix.get(1, 0).unwrap(), 3.0);
        assert_eq!(bdmatrix.get(1, 1).unwrap(), 4.0);
        assert_eq!(bdmatrix.get(2, 2).unwrap(), 5.0);

        // Off-diagonal elements should be zero
        assert_eq!(bdmatrix.get(0, 2).unwrap(), 0.0);
        assert_eq!(bdmatrix.get(2, 0).unwrap(), 0.0);
    }

    #[test]
    fn test_block_diagonal_matvec() {
        let block1 = array![[2.0, 0.0], [0.0, 3.0]];
        let block2 = array![[4.0]];
        let blocks = vec![block1, block2];
        let bdmatrix = BlockDiagonalMatrix::new(blocks).unwrap();

        let x = array![1.0, 2.0, 3.0];
        let result = bdmatrix.matvec(&x.view()).unwrap();

        // Expected: [2*1, 3*2, 4*3] = [2, 6, 12]
        assert_eq!(result, array![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_block_diagonal_determinant() {
        let block1 = array![[2.0, 0.0], [0.0, 3.0]];
        let block2 = array![[4.0]];
        let blocks = vec![block1, block2];
        let bdmatrix = BlockDiagonalMatrix::new(blocks).unwrap();

        let det = bdmatrix.determinant().unwrap();
        // Expected: det(block1) * det(block2) = 6 * 4 = 24
        assert!((det - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_block_diagonal_to_dense() {
        let block1 = array![[1.0, 2.0], [3.0, 4.0]];
        let block2 = array![[5.0]];
        let blocks = vec![block1, block2];
        let bdmatrix = BlockDiagonalMatrix::new(blocks).unwrap();

        let dense = bdmatrix.to_dense().unwrap();
        let expected = array![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]];

        assert_eq!(dense, expected);
    }
}
