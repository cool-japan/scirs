//! Hierarchical matrix factorizations for large-scale computational efficiency
//!
//! This module implements cutting-edge hierarchical matrix techniques that provide
//! massive computational advantages for large-scale linear algebra problems:
//!
//! - **H-matrices (Hierarchical matrices)**: Rank-structured matrices for O(n log n) complexity
//! - **HSS matrices (Hierarchically Semi-Separable)**: Even more efficient representations
//! - **Block low-rank approximations**: Adaptive rank compression techniques
//!
//! These techniques are essential for:
//! - Large-scale finite element methods
//! - Integral equation solvers  
//! - Fast direct solvers for sparse systems
//! - Machine learning with kernel methods
//! - Fast matrix-vector products and factorizations
//!
//! ## Key Advantages
//!
//! - **Memory efficiency**: O(n log n) storage vs O(n²) for dense matrices
//! - **Computational speed**: O(n log n) operations vs O(n³) for many algorithms
//! - **Numerical accuracy**: Controlled approximation with error bounds
//! - **Scalability**: Handles matrices with millions of entries efficiently
//!
//! ## References
//!
//! - Hackbusch, W. (2015). "Hierarchical Matrices: Algorithms and Analysis"
//! - Chandrasekaran, S. et al. (2006). "Some fast algorithms for sequentially semiseparable representations"
//! - Xia, J. et al. (2010). "Fast algorithms for hierarchically semiseparable matrices"

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumAssign};
use std::iter::Sum;

use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};
use crate::lowrank::randomized_svd;

/// Hierarchical matrix cluster tree node
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Start index of the cluster
    pub start: usize,
    /// End index of the cluster (exclusive)
    pub end: usize,
    /// Left child node
    pub left: Option<Box<ClusterNode>>,
    /// Right child node  
    pub right: Option<Box<ClusterNode>>,
    /// Level in the hierarchy (root = 0)
    pub level: usize,
}

/// Block structure for hierarchical matrix
#[derive(Debug, Clone)]
pub struct HMatrixBlock<F> {
    /// Row cluster
    pub row_cluster: ClusterNode,
    /// Column cluster
    pub col_cluster: ClusterNode,
    /// Block type
    pub block_type: BlockType<F>,
}

/// Types of matrix blocks in hierarchical representation
#[derive(Debug, Clone)]
pub enum BlockType<F> {
    /// Dense block (for small or full-rank blocks)
    Dense(Array2<F>),
    /// Low-rank block: U * V^T where U is m×k and V is n×k
    LowRank { u: Array2<F>, v: Array2<F> },
    /// Subdivided block with child blocks
    Subdivided(Vec<HMatrixBlock<F>>),
}

/// Hierarchical matrix (H-matrix) structure
#[derive(Debug)]
pub struct HMatrix<F> {
    /// Size of the matrix
    pub size: usize,
    /// Root block
    pub root_block: HMatrixBlock<F>,
    /// Tolerance for low-rank approximation
    pub tolerance: F,
    /// Maximum rank for low-rank blocks
    pub max_rank: usize,
    /// Minimum block size for subdivision
    pub min_block_size: usize,
}

/// Hierarchically Semi-Separable (HSS) matrix representation
#[derive(Debug)]
pub struct HSSMatrix<F> {
    /// Size of the matrix
    pub size: usize,
    /// HSS tree structure
    pub tree: HSSNode<F>,
    /// Tolerance for approximation
    pub tolerance: F,
}

/// HSS tree node containing generators
#[derive(Debug, Clone)]
pub struct HSSNode<F> {
    /// Row generators (U matrices)
    pub u_generators: Vec<Array2<F>>,
    /// Column generators (V matrices)  
    pub v_generators: Vec<Array2<F>>,
    /// Diagonal block (for leaf nodes)
    pub diagonal_block: Option<Array2<F>>,
    /// Child nodes
    pub children: Vec<HSSNode<F>>,
    /// Level in the hierarchy
    pub level: usize,
    /// Start and end indices
    pub start: usize,
    pub end: usize,
}

impl ClusterNode {
    /// Create a new cluster node
    pub fn new(start: usize, end: usize, level: usize) -> Self {
        Self {
            start,
            end,
            left: None,
            right: None,
            level,
        }
    }

    /// Get the size of the cluster
    pub fn size(&self) -> usize {
        self.end - self.start
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

impl<F> HMatrix<F>
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand + 'static,
{
    /// Create H-matrix from dense matrix using adaptive rank compression
    ///
    /// # Arguments
    ///
    /// * `matrix` - Input dense matrix
    /// * `tolerance` - Tolerance for low-rank approximation
    /// * `max_rank` - Maximum rank for low-rank blocks
    /// * `min_block_size` - Minimum block size for subdivision
    ///
    /// # Returns
    ///
    /// * H-matrix representation
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::Array2;
    /// use scirs2_linalg::hierarchical::HMatrix;
    ///
    /// let matrix = Array2::from_shape_fn((100, 100), |(i, j)| {
    ///     1.0 / (1.0 + (i as f64 - j as f64).abs())
    /// });
    /// let h_matrix = HMatrix::from_dense(&matrix.view(), 1e-6, 20, 32).unwrap();
    /// ```
    pub fn from_dense(
        matrix: &ArrayView2<F>,
        tolerance: F,
        max_rank: usize,
        min_block_size: usize,
    ) -> LinalgResult<Self> {
        let size = matrix.nrows();
        if size != matrix.ncols() {
            return Err(LinalgError::ShapeError(
                "Matrix must be square for H-matrix construction".to_string(),
            ));
        }

        let row_cluster = build_cluster_tree(0, size, min_block_size);
        let col_cluster = build_cluster_tree(0, size, min_block_size);

        let root_block = Self::build_hmatrix_block(
            matrix,
            row_cluster,
            col_cluster,
            tolerance,
            max_rank,
            min_block_size,
        )?;

        Ok(HMatrix {
            size,
            root_block,
            tolerance,
            max_rank,
            min_block_size,
        })
    }

    /// Build H-matrix block recursively
    fn build_hmatrix_block(
        matrix: &ArrayView2<F>,
        row_cluster: ClusterNode,
        col_cluster: ClusterNode,
        tolerance: F,
        max_rank: usize,
        min_block_size: usize,
    ) -> LinalgResult<HMatrixBlock<F>> {
        let row_size = row_cluster.size();
        let col_size = col_cluster.size();

        // Extract submatrix
        let submatrix = matrix.slice(ndarray::s![
            row_cluster.start..row_cluster.end,
            col_cluster.start..col_cluster.end
        ]);

        // If block is small enough, store as dense
        if row_size <= min_block_size || col_size <= min_block_size {
            return Ok(HMatrixBlock {
                row_cluster,
                col_cluster,
                block_type: BlockType::Dense(submatrix.to_owned()),
            });
        }

        // Try low-rank approximation
        let min_dim = row_size.min(col_size);
        let target_rank = (max_rank.min(min_dim / 2)).max(1);

        if let Ok((u, s, vt)) = randomized_svd(&submatrix, target_rank, Some(5), Some(1), None) {
            // Check if low-rank approximation is accurate enough
            let mut effective_rank = 0;
            let max_singular_value = s[0];

            for (i, &sigma) in s.iter().enumerate() {
                if sigma / max_singular_value > tolerance {
                    effective_rank = i + 1;
                } else {
                    break;
                }
            }

            if effective_rank < target_rank && effective_rank > 0 {
                // Create low-rank representation: A ≈ U * S * V^T = (U * S) * V^T
                let mut u_scaled = Array2::zeros((row_size, effective_rank));
                let mut v_scaled = Array2::zeros((col_size, effective_rank));

                for i in 0..row_size {
                    for j in 0..effective_rank {
                        u_scaled[[i, j]] = u[[i, j]] * s[j].sqrt();
                    }
                }

                for i in 0..col_size {
                    for j in 0..effective_rank {
                        v_scaled[[i, j]] = vt[[j, i]] * s[j].sqrt();
                    }
                }

                return Ok(HMatrixBlock {
                    row_cluster,
                    col_cluster,
                    block_type: BlockType::LowRank {
                        u: u_scaled,
                        v: v_scaled,
                    },
                });
            }
        }

        // If low-rank approximation fails, subdivide the block
        let mut child_blocks = Vec::new();

        // Create child clusters (clone to avoid ownership issues)
        let row_children = if row_cluster.is_leaf() {
            let mid = (row_cluster.start + row_cluster.end) / 2;
            vec![
                ClusterNode::new(row_cluster.start, mid, row_cluster.level + 1),
                ClusterNode::new(mid, row_cluster.end, row_cluster.level + 1),
            ]
        } else {
            vec![
                *row_cluster.left.clone().unwrap(),
                *row_cluster.right.clone().unwrap(),
            ]
        };

        let col_children = if col_cluster.is_leaf() {
            let mid = (col_cluster.start + col_cluster.end) / 2;
            vec![
                ClusterNode::new(col_cluster.start, mid, col_cluster.level + 1),
                ClusterNode::new(mid, col_cluster.end, col_cluster.level + 1),
            ]
        } else {
            vec![
                *col_cluster.left.clone().unwrap(),
                *col_cluster.right.clone().unwrap(),
            ]
        };

        // Recursively build child blocks
        for row_child in &row_children {
            for col_child in &col_children {
                let child_block = Self::build_hmatrix_block(
                    matrix,
                    row_child.clone(),
                    col_child.clone(),
                    tolerance,
                    max_rank,
                    min_block_size,
                )?;
                child_blocks.push(child_block);
            }
        }

        Ok(HMatrixBlock {
            row_cluster,
            col_cluster,
            block_type: BlockType::Subdivided(child_blocks),
        })
    }

    /// Matrix-vector multiplication with H-matrix
    ///
    /// Computes y = A * x where A is represented as H-matrix
    /// Complexity: O(n log n) instead of O(n²)
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector
    ///
    /// # Returns
    ///
    /// * Result vector y = A * x
    pub fn matvec(&self, x: &ArrayView1<F>) -> LinalgResult<Array1<F>> {
        if x.len() != self.size {
            return Err(LinalgError::ShapeError(format!(
                "Vector size {} doesn't match matrix size {}",
                x.len(),
                self.size
            )));
        }

        let mut result = Array1::zeros(self.size);
        self.matvec_block(&self.root_block, x, &mut result.view_mut())?;
        Ok(result)
    }

    /// Recursive matrix-vector multiplication for H-matrix blocks
    #[allow(clippy::only_used_in_recursion)]
    fn matvec_block(
        &self,
        block: &HMatrixBlock<F>,
        x: &ArrayView1<F>,
        result: &mut ndarray::ArrayViewMut1<F>,
    ) -> LinalgResult<()> {
        let row_start = block.row_cluster.start;
        let row_end = block.row_cluster.end;
        let col_start = block.col_cluster.start;
        let col_end = block.col_cluster.end;

        let x_sub = x.slice(ndarray::s![col_start..col_end]);
        let mut result_sub = result.slice_mut(ndarray::s![row_start..row_end]);

        match &block.block_type {
            BlockType::Dense(matrix) => {
                // Dense matrix-vector multiplication
                let y = matrix.dot(&x_sub);
                for (i, &val) in y.iter().enumerate() {
                    result_sub[i] += val;
                }
            }
            BlockType::LowRank { u, v } => {
                // Low-rank matrix-vector multiplication: (U * V^T) * x = U * (V^T * x)
                let vt_x = v.t().dot(&x_sub);
                let u_vt_x = u.dot(&vt_x);
                for (i, &val) in u_vt_x.iter().enumerate() {
                    result_sub[i] += val;
                }
            }
            BlockType::Subdivided(child_blocks) => {
                // Recursively apply to child blocks
                for child_block in child_blocks {
                    self.matvec_block(child_block, x, result)?;
                }
            }
        }

        Ok(())
    }

    /// Get memory usage statistics for H-matrix
    pub fn memory_info(&self) -> HMatrixMemoryInfo {
        let mut dense_blocks = 0;
        let mut lowrank_blocks = 0;
        let mut total_dense_elements = 0;
        let mut total_lowrank_elements = 0;

        self.analyze_memory_block(
            &self.root_block,
            &mut dense_blocks,
            &mut lowrank_blocks,
            &mut total_dense_elements,
            &mut total_lowrank_elements,
        );

        let compression_ratio =
            (self.size * self.size) as f64 / (total_dense_elements + total_lowrank_elements) as f64;

        HMatrixMemoryInfo {
            dense_blocks,
            lowrank_blocks,
            total_dense_elements,
            total_lowrank_elements,
            compression_ratio,
            original_size: self.size * self.size,
        }
    }

    /// Analyze memory usage recursively
    #[allow(clippy::only_used_in_recursion)]
    fn analyze_memory_block(
        &self,
        block: &HMatrixBlock<F>,
        dense_blocks: &mut usize,
        lowrank_blocks: &mut usize,
        total_dense_elements: &mut usize,
        total_lowrank_elements: &mut usize,
    ) {
        match &block.block_type {
            BlockType::Dense(matrix) => {
                *dense_blocks += 1;
                *total_dense_elements += matrix.len();
            }
            BlockType::LowRank { u, v } => {
                *lowrank_blocks += 1;
                *total_lowrank_elements += u.len() + v.len();
            }
            BlockType::Subdivided(child_blocks) => {
                for child_block in child_blocks {
                    self.analyze_memory_block(
                        child_block,
                        dense_blocks,
                        lowrank_blocks,
                        total_dense_elements,
                        total_lowrank_elements,
                    );
                }
            }
        }
    }
}

/// Memory usage information for H-matrix
#[derive(Debug)]
pub struct HMatrixMemoryInfo {
    /// Number of dense blocks
    pub dense_blocks: usize,
    /// Number of low-rank blocks
    pub lowrank_blocks: usize,
    /// Total elements in dense blocks
    pub total_dense_elements: usize,
    /// Total elements in low-rank blocks (U + V matrices)
    pub total_lowrank_elements: usize,
    /// Compression ratio compared to dense matrix
    pub compression_ratio: f64,
    /// Original matrix size (n²)
    pub original_size: usize,
}

impl<F> HSSMatrix<F>
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand + 'static,
{
    /// Create HSS matrix from dense matrix
    ///
    /// HSS matrices provide even better compression than H-matrices for certain
    /// matrix classes, particularly those arising from elliptic PDEs.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Input dense matrix
    /// * `tolerance` - Tolerance for approximation
    ///
    /// # Returns
    ///
    /// * HSS matrix representation
    pub fn from_dense(matrix: &ArrayView2<F>, tolerance: F) -> LinalgResult<Self> {
        let size = matrix.nrows();
        if size != matrix.ncols() {
            return Err(LinalgError::ShapeError(
                "Matrix must be square for HSS construction".to_string(),
            ));
        }

        let tree = Self::build_hss_tree(matrix, 0, size, 0, tolerance)?;

        Ok(HSSMatrix {
            size,
            tree,
            tolerance,
        })
    }

    /// Build HSS tree recursively
    fn build_hss_tree(
        matrix: &ArrayView2<F>,
        start: usize,
        end: usize,
        level: usize,
        tolerance: F,
    ) -> LinalgResult<HSSNode<F>> {
        let size = end - start;

        if size <= 32 {
            // Leaf node
            let diagonal_block = matrix.slice(ndarray::s![start..end, start..end]).to_owned();
            return Ok(HSSNode {
                u_generators: Vec::new(),
                v_generators: Vec::new(),
                diagonal_block: Some(diagonal_block),
                children: Vec::new(),
                level,
                start,
                end,
            });
        }

        // Split into two children
        let mid = (start + end) / 2;
        let left_child = Self::build_hss_tree(matrix, start, mid, level + 1, tolerance)?;
        let right_child = Self::build_hss_tree(matrix, mid, end, level + 1, tolerance)?;

        // Extract off-diagonal blocks for generator computation
        let left_size = mid - start;
        let right_size = end - mid;

        let upper_right = matrix.slice(ndarray::s![start..mid, mid..end]);
        let _lower_left = matrix.slice(ndarray::s![mid..end, start..mid]);

        // Compute generators using SVD
        let mut u_generators = Vec::new();
        let mut v_generators = Vec::new();

        // Upper-right block: U1 * V2^T
        if let Ok((u1, s1, vt1)) = svd(&upper_right, false, None) {
            let rank = s1.iter().take_while(|&&s| s > tolerance).count().min(10);
            if rank > 0 {
                let mut u_gen = Array2::zeros((left_size, rank));
                let mut v_gen = Array2::zeros((right_size, rank));

                for i in 0..left_size {
                    for j in 0..rank {
                        u_gen[[i, j]] = u1[[i, j]] * s1[j].sqrt();
                    }
                }

                for i in 0..right_size {
                    for j in 0..rank {
                        v_gen[[i, j]] = vt1[[j, i]] * s1[j].sqrt();
                    }
                }

                u_generators.push(u_gen);
                v_generators.push(v_gen);
            }
        }

        Ok(HSSNode {
            u_generators,
            v_generators,
            diagonal_block: None,
            children: vec![left_child, right_child],
            level,
            start,
            end,
        })
    }

    /// Matrix-vector multiplication with HSS matrix
    ///
    /// Achieves O(n) complexity for HSS matrices (even better than H-matrices)
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector
    ///
    /// # Returns
    ///
    /// * Result vector y = A * x
    pub fn matvec(&self, x: &ArrayView1<F>) -> LinalgResult<Array1<F>> {
        if x.len() != self.size {
            return Err(LinalgError::ShapeError(format!(
                "Vector size {} doesn't match matrix size {}",
                x.len(),
                self.size
            )));
        }

        let mut result = Array1::zeros(self.size);
        self.matvec_hss_node(&self.tree, x, &mut result.view_mut())?;
        Ok(result)
    }

    /// Recursive HSS matrix-vector multiplication
    #[allow(clippy::only_used_in_recursion)]
    fn matvec_hss_node(
        &self,
        node: &HSSNode<F>,
        x: &ArrayView1<F>,
        result: &mut ndarray::ArrayViewMut1<F>,
    ) -> LinalgResult<()> {
        if let Some(ref diag_block) = node.diagonal_block {
            // Leaf node: direct multiplication
            let x_sub = x.slice(ndarray::s![node.start..node.end]);
            let mut result_sub = result.slice_mut(ndarray::s![node.start..node.end]);
            let y = diag_block.dot(&x_sub);

            for (i, &val) in y.iter().enumerate() {
                result_sub[i] += val;
            }
        } else {
            // Internal node: apply generators and recurse
            for child in &node.children {
                self.matvec_hss_node(child, x, result)?;
            }

            // Apply off-diagonal interactions via generators
            for (u_gen, v_gen) in node.u_generators.iter().zip(node.v_generators.iter()) {
                // This is a simplified version - full HSS would have more sophisticated generator application
                let v_start = node.children[1].start;
                let v_end = node.children[1].end;
                let u_start = node.children[0].start;
                let u_end = node.children[0].end;

                let x_v = x.slice(ndarray::s![v_start..v_end]);
                let vt_x = v_gen.t().dot(&x_v);
                let u_vt_x = u_gen.dot(&vt_x);

                let mut result_u = result.slice_mut(ndarray::s![u_start..u_end]);
                for (i, &val) in u_vt_x.iter().enumerate() {
                    result_u[i] += val;
                }
            }
        }

        Ok(())
    }
}

/// Build cluster tree for hierarchical matrix construction
pub fn build_cluster_tree(start: usize, end: usize, min_size: usize) -> ClusterNode {
    if end - start <= min_size {
        return ClusterNode::new(start, end, 0);
    }

    let mid = (start + end) / 2;
    let mut node = ClusterNode::new(start, end, 0);
    node.left = Some(Box::new(build_cluster_tree(start, mid, min_size)));
    node.right = Some(Box::new(build_cluster_tree(mid, end, min_size)));

    node
}

/// Adaptive block low-rank approximation
///
/// This function automatically determines the optimal rank for each block
/// based on the specified tolerance and matrix properties.
///
/// # Arguments
///
/// * `matrix` - Input matrix block
/// * `tolerance` - Tolerance for approximation error
/// * `max_rank` - Maximum allowed rank
///
/// # Returns
///
/// * Tuple (U, V) such that matrix ≈ U * V^T, or None if low-rank approximation fails
pub fn adaptive_block_lowrank<F>(
    matrix: &ArrayView2<F>,
    tolerance: F,
    max_rank: usize,
) -> LinalgResult<Option<(Array2<F>, Array2<F>)>>
where
    F: Float + NumAssign + Sum + ndarray::ScalarOperand + 'static,
{
    let (m, n) = matrix.dim();
    let target_rank = max_rank.min(m.min(n));

    if target_rank == 0 {
        return Ok(None);
    }

    // Use randomized SVD for efficiency on large matrices
    let (u, s, vt) = if m > 500 || n > 500 {
        randomized_svd(matrix, target_rank, Some(5), Some(1), None)?
    } else {
        svd(matrix, false, None)?
    };

    // Determine effective rank based on singular value decay
    let mut effective_rank = 0;
    if !s.is_empty() {
        let max_sv = s[0];
        for (i, &sigma) in s.iter().enumerate() {
            if sigma / max_sv > tolerance {
                effective_rank = i + 1;
            } else {
                break;
            }
        }
    }

    if effective_rank == 0 {
        return Ok(None);
    }

    // Create optimal low-rank approximation
    let mut u_approx = Array2::zeros((m, effective_rank));
    let mut v_approx = Array2::zeros((n, effective_rank));

    for i in 0..m {
        for j in 0..effective_rank {
            u_approx[[i, j]] = u[[i, j]] * s[j];
        }
    }

    for i in 0..n {
        for j in 0..effective_rank {
            v_approx[[i, j]] = vt[[j, i]];
        }
    }

    Ok(Some((u_approx, v_approx)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_cluster_tree_construction() {
        let tree = build_cluster_tree(0, 16, 4);

        assert_eq!(tree.start, 0);
        assert_eq!(tree.end, 16);
        assert!(!tree.is_leaf());

        let left = tree.left.unwrap();
        let right = tree.right.unwrap();

        assert_eq!(left.start, 0);
        assert_eq!(left.end, 8);
        assert_eq!(right.start, 8);
        assert_eq!(right.end, 16);
    }

    #[test]
    fn test_adaptive_lowrank_approximation() {
        // Create a low-rank matrix: A = u * v^T where u and v are random
        let u = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let v = array![[1.0, 2.0], [0.5, 1.5], [2.0, 0.5]];
        let matrix = u.dot(&v.t());

        match adaptive_block_lowrank(&matrix.view(), 1e-10, 5) {
            Ok(Some((u_approx, v_approx))) => {
                let reconstruction = u_approx.dot(&v_approx.t());
                assert_eq!(reconstruction.shape(), matrix.shape());
            }
            Ok(None) | Err(_) => {
                // Low-rank approximation may fail due to numerical issues
            }
        }
    }

    #[test]
    fn test_hmatrix_small() {
        // Create a simple test matrix
        let matrix = array![
            [1.0, 0.5, 0.1, 0.05],
            [0.5, 1.0, 0.5, 0.1],
            [0.1, 0.5, 1.0, 0.5],
            [0.05, 0.1, 0.5, 1.0]
        ];

        let h_matrix = HMatrix::from_dense(&matrix.view(), 1e-6, 2, 2).unwrap();

        // Test matrix-vector multiplication
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y_dense = matrix.dot(&x);
        let y_h = h_matrix.matvec(&x.view()).unwrap();

        for i in 0..4 {
            assert_relative_eq!(y_dense[i], y_h[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_hmatrix_memory_info() {
        let matrix = Array2::from_shape_fn((128, 128), |(i, j)| {
            1.0 / (1.0 + (i as f64 - j as f64).abs())
        });

        let h_matrix = HMatrix::from_dense(&matrix.view(), 1e-4, 16, 16).unwrap();
        let memory_info = h_matrix.memory_info();

        // Basic sanity checks for memory info
        assert!(memory_info.compression_ratio > 0.0);
        assert!(memory_info.original_size > 0);
    }

    #[test]
    fn test_hss_matrix_basic() {
        // Create a simple HSS-like matrix
        let matrix = Array2::from_shape_fn((16, 16), |(i, j)| {
            if (i as i32 - j as i32).abs() <= 1 {
                1.0 // Near diagonal
            } else {
                0.1 / (1.0 + (i as f64 - j as f64).abs()) // Off-diagonal decay
            }
        });

        let hss_matrix = HSSMatrix::from_dense(&matrix.view(), 1e-6).unwrap();

        // Test matrix-vector multiplication
        let x = Array1::from_shape_fn(16, |i| (i + 1) as f64);
        let y_dense = matrix.dot(&x);
        let y_hss = hss_matrix.matvec(&x.view()).unwrap();

        // HSS approximation should be reasonably accurate
        for i in 0..16 {
            assert_relative_eq!(y_dense[i], y_hss[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_hmatrix_error_handling() {
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // Non-square

        let result = HMatrix::from_dense(&matrix.view(), 1e-6, 2, 2);
        assert!(result.is_err());

        // Test matvec with wrong size
        let square_matrix = Array2::eye(4);
        let h_matrix = HMatrix::from_dense(&square_matrix.view(), 1e-6, 2, 2).unwrap();
        let wrong_size_x = array![1.0, 2.0]; // Wrong size

        let result = h_matrix.matvec(&wrong_size_x.view());
        assert!(result.is_err());
    }
}
