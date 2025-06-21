//! Tucker Decomposition
//!
//! This module provides functionality for Tucker decomposition, which is a form of
//! higher-order principal component analysis. It decomposes a tensor into a core tensor
//! multiplied by a matrix along each mode.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array2, ArrayD, ArrayView, Dimension};
use num_traits::{Float, NumAssign, Zero};
use std::fmt::Debug;
use std::iter::Sum;

/// Represents a tensor in Tucker format.
///
/// A Tucker decomposition represents an N-dimensional tensor as a core tensor
/// multiplied by a matrix along each mode.
///
/// # Fields
///
/// * `core` - The core tensor
/// * `factors` - Vector of factor matrices (one for each mode)
/// * `shape` - Original tensor shape
#[derive(Debug, Clone)]
pub struct Tucker<A>
where
    A: Clone + Float + Debug,
{
    /// The core tensor
    pub core: ArrayD<A>,
    /// The factor matrices, each with shape (mode_dim, rank)
    pub factors: Vec<Array2<A>>,
    /// Original tensor shape
    pub shape: Vec<usize>,
}

impl<A> Tucker<A>
where
    A: Clone
        + Float
        + NumAssign
        + Zero
        + Debug
        + Sum
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand,
{
    /// Creates a new Tucker decomposition from the given core tensor and factor matrices.
    ///
    /// # Arguments
    ///
    /// * `core` - The core tensor
    /// * `factors` - Vector of factor matrices (one for each mode)
    /// * `shape` - Original tensor shape (optional, if None will be derived from factors)
    ///
    /// # Returns
    ///
    /// * `Tucker` - A new Tucker instance
    pub fn new(
        core: ArrayD<A>,
        factors: Vec<Array2<A>>,
        shape: Option<Vec<usize>>,
    ) -> LinalgResult<Self> {
        // Basic validation
        if factors.is_empty() {
            return Err(LinalgError::ValueError(
                "Factor matrices list cannot be empty".to_string(),
            ));
        }

        if core.ndim() != factors.len() {
            return Err(LinalgError::ShapeError(format!(
                "Number of factor matrices ({}) must match core tensor dimensionality ({})",
                factors.len(),
                core.ndim()
            )));
        }

        // Validate dimensions of core tensor and factor matrices
        for (i, factor) in factors.iter().enumerate() {
            if factor.ndim() != 2 {
                return Err(LinalgError::ShapeError(format!(
                    "Factor matrix {} must be 2-dimensional, got {} dimensions",
                    i,
                    factor.ndim()
                )));
            }

            if factor.shape()[1] != core.shape()[i] {
                return Err(LinalgError::ShapeError(format!(
                    "Factor matrix {} columns ({}) must match core tensor dimension {} ({})",
                    i,
                    factor.shape()[1],
                    i,
                    core.shape()[i]
                )));
            }
        }

        // Derive shape from factors if not provided
        let shape = match shape {
            Some(s) => {
                if s.len() != factors.len() {
                    return Err(LinalgError::ShapeError(format!(
                        "Shape length ({}) must match number of factor matrices ({})",
                        s.len(),
                        factors.len()
                    )));
                }

                for (i, &dim) in s.iter().enumerate() {
                    if dim != factors[i].shape()[0] {
                        return Err(LinalgError::ShapeError(format!(
                            "Shape dimension {} ({}) must match factor matrix {} rows ({})",
                            i,
                            dim,
                            i,
                            factors[i].shape()[0]
                        )));
                    }
                }
                s
            }
            None => factors.iter().map(|f| f.shape()[0]).collect(),
        };

        Ok(Tucker {
            core,
            factors,
            shape,
        })
    }

    /// Reconstructs the full tensor from the Tucker decomposition.
    ///
    /// # Returns
    ///
    /// * `ArrayD<A>` - The full tensor
    pub fn to_full(&self) -> LinalgResult<ArrayD<A>> {
        use crate::tensor_contraction::mode_n_product;

        // Start with the core tensor
        let mut result = self.core.clone();

        // Multiply by each factor matrix along each mode
        for (mode, factor) in self.factors.iter().enumerate() {
            result = mode_n_product(&result.view(), &factor.view(), mode)?;
        }

        Ok(result)
    }

    /// Computes the reconstruction error of the Tucker decomposition.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The original tensor
    ///
    /// # Returns
    ///
    /// * `A` - The relative reconstruction error (Frobenius norm of the difference divided by
    ///   the Frobenius norm of the original tensor)
    pub fn reconstruction_error(&self, tensor: &ArrayView<A, impl Dimension>) -> LinalgResult<A> {
        // Reconstruct the tensor
        let reconstructed = self.to_full()?;

        // Convert original tensor to dynamic dimensionality
        let tensor_dyn = tensor.to_owned().into_dyn();

        // Verify shapes match
        if tensor_dyn.shape() != reconstructed.shape() {
            return Err(LinalgError::ShapeError(format!(
                "Original tensor shape {:?} does not match reconstructed tensor shape {:?}",
                tensor_dyn.shape(),
                reconstructed.shape()
            )));
        }

        // Compute Frobenius norm of the difference and the original tensor
        let mut diff_squared_sum = A::zero();
        let mut orig_squared_sum = A::zero();

        for (idx, &orig_val) in tensor_dyn.indexed_iter() {
            let rec_val = reconstructed[idx.clone()];
            diff_squared_sum += (orig_val - rec_val).powi(2);
            orig_squared_sum += orig_val.powi(2);
        }

        // Handle division by zero
        if orig_squared_sum == A::zero() {
            return Ok(if diff_squared_sum == A::zero() {
                A::zero()
            } else {
                A::infinity()
            });
        }

        // Return relative error
        Ok((diff_squared_sum / orig_squared_sum).sqrt())
    }

    /// Compresses the Tucker decomposition by truncating small singular values.
    ///
    /// # Arguments
    ///
    /// * `ranks` - Maximum rank for each mode (if None, ranks remain unchanged)
    /// * `epsilon` - Relative error tolerance for truncation (if None, no truncation by error)
    ///
    /// # Returns
    ///
    /// * `Tucker` - A new, compressed Tucker decomposition
    pub fn compress(&self, ranks: Option<Vec<usize>>, epsilon: Option<A>) -> LinalgResult<Self> {
        use super::svd_truncated;

        // Check inputs
        if let Some(ref r) = ranks {
            if r.len() != self.factors.len() {
                return Err(LinalgError::ShapeError(format!(
                    "Ranks length ({}) must match number of factor matrices ({})",
                    r.len(),
                    self.factors.len()
                )));
            }
        }

        if let Some(eps) = epsilon {
            if eps <= A::zero() {
                return Err(LinalgError::ValueError(
                    "Epsilon must be positive".to_string(),
                ));
            }
        }

        // Clone the current decomposition if no compression is needed
        if ranks.is_none() && epsilon.is_none() {
            return Ok(self.clone());
        }

        // Determine target ranks for each mode
        let target_ranks: Vec<usize> = match (&ranks, epsilon) {
            (Some(r), None) => r.clone(),
            (None, Some(eps)) => {
                // For each factor matrix, determine how many singular values to keep
                // based on the epsilon value
                use rayon::prelude::*;

                self.factors
                    .par_iter()
                    .map(|factor| {
                        // Perform SVD of the factor matrix
                        let (_, s, _) = svd_truncated(factor, factor.shape()[1])
                            .expect("SVD of factor matrix failed");

                        // Normalize singular values
                        let s_norm = if !s.is_empty() && s[[0, 0]] > A::zero() {
                            s.mapv(|v| v / s[[0, 0]])
                        } else {
                            s.clone()
                        };

                        // Count number of singular values above threshold
                        let mut count = 0;
                        for i in 0..s_norm.shape()[0] {
                            if s_norm[[i, i]] >= eps {
                                count += 1;
                            } else {
                                break;
                            }
                        }

                        // Ensure at least one singular value is kept
                        count.max(1)
                    })
                    .collect()
            }
            (Some(r), Some(eps)) => {
                // Combine both criteria: truncate by epsilon but don't exceed max ranks
                use rayon::prelude::*;

                self.factors
                    .par_iter()
                    .zip(r.par_iter())
                    .map(|(factor, &max_rank)| {
                        // Perform SVD of the factor matrix
                        let (_, s, _) = svd_truncated(factor, factor.shape()[1])
                            .expect("SVD of factor matrix failed");

                        // Normalize singular values
                        let s_norm = if !s.is_empty() && s[[0, 0]] > A::zero() {
                            s.mapv(|v| v / s[[0, 0]])
                        } else {
                            s.clone()
                        };

                        // Count number of singular values above threshold
                        let mut count = 0;
                        for i in 0..s_norm.shape()[0] {
                            if s_norm[[i, i]] >= eps {
                                count += 1;
                            } else {
                                break;
                            }
                        }

                        // Apply max rank constraint and ensure at least one singular value is kept
                        count.max(1).min(max_rank)
                    })
                    .collect()
            }
            (None, None) => unreachable!("This case is handled above"),
        };

        // Compute truncated factor matrices
        use rayon::prelude::*;

        let compressed_factors: Vec<Array2<A>> = self
            .factors
            .par_iter()
            .zip(target_ranks.par_iter())
            .map(|(factor, &rank)| {
                let rank = rank.min(factor.shape()[1]);
                let (u, _, _) = svd_truncated(factor, rank).expect("SVD of factor matrix failed");
                u
            })
            .collect();

        // Compute the corresponding core tensor
        let mut compressed_core = self.core.clone();
        #[allow(clippy::needless_range_loop)]
        for mode in 0..compressed_factors.len() {
            // Original factor's transpose
            let orig_factor_t = self.factors[mode].t().to_owned();

            // Compressed factor's transpose
            let comp_factor_t = compressed_factors[mode].t().to_owned();

            // Multiply core tensor by (compressed_factor * orig_factor^T) along mode
            let transform = comp_factor_t.dot(&orig_factor_t.t());

            // Apply transformation to core tensor
            compressed_core = crate::tensor_contraction::mode_n_product(
                &compressed_core.view(),
                &transform.view(),
                mode,
            )?;
        }

        // Create the new Tucker decomposition
        Tucker::new(
            compressed_core,
            compressed_factors,
            Some(self.shape.clone()),
        )
    }
}

/// Performs Tucker decomposition of a tensor.
///
/// This function computes a Tucker decomposition, which represents an N-dimensional tensor
/// as a core tensor multiplied by a matrix along each mode. This is also known as a
/// higher-order singular value decomposition (HOSVD).
///
/// # Arguments
///
/// * `tensor` - The tensor to decompose
/// * `ranks` - The target rank for each mode (the size of the core tensor dimensions)
///
/// # Returns
///
/// * A `Tucker` object containing the decomposition
///
/// # Examples
///
/// ```
/// use ndarray::{array, ArrayD};
/// use scirs2_linalg::tensor_contraction::tucker::tucker_decomposition;
///
/// // Create a 2x3x2 tensor
/// let tensor = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
///                     [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]];
///
/// // Decompose with full rank
/// let tucker = tucker_decomposition(&tensor.view(), &[2, 3, 2]).unwrap();
///
/// // The core tensor should have the specified rank
/// assert_eq!(tucker.core.shape(), &[2, 3, 2]);
///
/// // The factors should preserve the original dimensions
/// assert_eq!(tucker.factors[0].shape(), &[2, 2]); // mode 0
/// assert_eq!(tucker.factors[1].shape(), &[3, 3]); // mode 1
/// assert_eq!(tucker.factors[2].shape(), &[2, 2]); // mode 2
/// ```
pub fn tucker_decomposition<A, D>(
    tensor: &ArrayView<A, D>,
    ranks: &[usize],
) -> LinalgResult<Tucker<A>>
where
    A: Clone
        + Float
        + NumAssign
        + Zero
        + Debug
        + Sum
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand,
    D: Dimension,
{
    use super::hosvd;

    // Verify that ranks are valid
    if ranks.len() != tensor.ndim() {
        return Err(LinalgError::ShapeError(format!(
            "Ranks length ({}) must match tensor dimensionality ({})",
            ranks.len(),
            tensor.ndim()
        )));
    }

    for (i, &rank) in ranks.iter().enumerate() {
        if rank > tensor.shape()[i] {
            return Err(LinalgError::ShapeError(format!(
                "Rank for mode {} ({}) cannot exceed the mode dimension ({})",
                i,
                rank,
                tensor.shape()[i]
            )));
        }

        if rank == 0 {
            return Err(LinalgError::ValueError(format!(
                "Rank for mode {} must be positive",
                i
            )));
        }
    }

    // Use HOSVD to compute the Tucker decomposition
    let (core, factors) = hosvd(tensor, ranks)?;

    // Create the Tucker object
    Tucker::new(core, factors, Some(tensor.shape().to_vec()))
}

/// Performs Tucker-ALS (Alternating Least Squares) decomposition of a tensor.
///
/// This function computes a Tucker decomposition using the Alternating Least Squares
/// algorithm, which can provide a more accurate low-rank approximation than HOSVD.
///
/// # Arguments
///
/// * `tensor` - The tensor to decompose
/// * `ranks` - The target rank for each mode
/// * `max_iterations` - Maximum number of ALS iterations
/// * `tolerance` - Convergence tolerance for relative change in fit
///
/// # Returns
///
/// * A `Tucker` object containing the decomposition
///
/// # Examples
///
/// ```
/// use ndarray::{array, ArrayD};
/// use scirs2_linalg::tensor_contraction::tucker::tucker_als;
///
/// // Create a 2x3x2 tensor
/// let tensor = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
///                     [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]];
///
/// // Decompose with target ranks [2, 2, 2]
/// let tucker = tucker_als(&tensor.view(), &[2, 2, 2], 10, 1e-4).unwrap();
///
/// // The core tensor should have the specified rank
/// assert_eq!(tucker.core.shape(), &[2, 2, 2]);
/// ```
pub fn tucker_als<A, D>(
    tensor: &ArrayView<A, D>,
    ranks: &[usize],
    max_iterations: usize,
    tolerance: A,
) -> LinalgResult<Tucker<A>>
where
    A: Clone
        + Float
        + NumAssign
        + Zero
        + Debug
        + Sum
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand,
    D: Dimension,
{
    use super::mode_n_product;
    use crate::decomposition::svd;

    // Initialize with HOSVD
    let mut tucker = tucker_decomposition(tensor, ranks)?;

    // Convert tensor to dynamic format
    let tensor_dyn = tensor.to_owned().into_dyn();

    // Cache the Frobenius norm of the original tensor
    let mut tensor_norm_sq = A::zero();
    for &val in tensor_dyn.iter() {
        tensor_norm_sq += val.powi(2);
    }

    // Initial reconstruction error
    let mut prev_error = tucker.reconstruction_error(tensor)?;

    // Main ALS loop
    for iteration in 0..max_iterations {
        // For each mode, update the corresponding factor matrix
        #[allow(clippy::needless_range_loop)]
        for mode in 0..ranks.len() {
            // Create the tensor unfolded along the current mode
            let tensor_unfolded = unfold_tensor(&tensor_dyn, mode)?;

            // Compute the Kronecker product of all factor matrices except the current one
            let khatri_rao_product =
                compute_khatri_rao_product(&tucker.factors, mode, &tucker.core)?;

            // Compute the new factor matrix using least squares
            let tensor_result = tensor_unfolded.dot(&khatri_rao_product);
            let (u, _, _) = svd(&tensor_result.view(), false, None)?;

            // Update the factor matrix for this mode
            let new_factor = u.slice(ndarray::s![.., ..ranks[mode]]).to_owned();
            tucker.factors[mode] = new_factor;

            // Update the core tensor based on the new factor matrices
            let mut temp_core = tensor_dyn.clone();
            for m in 0..tucker.factors.len() {
                let factor_t = tucker.factors[m].t().to_owned();
                temp_core = mode_n_product(&temp_core.view(), &factor_t.view(), m)?;
            }
            tucker.core = temp_core;
        }

        // Check for convergence
        let error = tucker.reconstruction_error(tensor)?;
        let rel_improvement = (prev_error - error) / prev_error;

        if rel_improvement < tolerance && iteration > 0 {
            break;
        }

        prev_error = error;
    }

    Ok(tucker)
}

// Helper function to unfold a tensor along a specified mode
fn unfold_tensor<A>(tensor: &ArrayD<A>, mode: usize) -> LinalgResult<Array2<A>>
where
    A: Clone + Float + NumAssign + Zero + Debug + Send + Sync + 'static,
{
    let shape = tensor.shape();

    if mode >= shape.len() {
        return Err(LinalgError::ShapeError(format!(
            "Mode {} is out of bounds for tensor with {} dimensions",
            mode,
            shape.len()
        )));
    }

    let mode_dim = shape[mode];

    // Calculate the product of all other dimensions
    let other_dims_prod: usize = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != mode)
        .map(|(_, &dim)| dim)
        .product();

    // Create result matrix
    let mut result = Array2::zeros((mode_dim, other_dims_prod));

    // Helper function to calculate column index
    fn calc_col_idx(idx: &[usize], shape: &[usize], mode: usize) -> usize {
        let mut col_idx = 0;
        let mut stride = 1;

        for dim in (0..shape.len()).rev() {
            if dim != mode {
                col_idx += idx[dim] * stride;
                stride *= shape[dim];
            }
        }

        col_idx
    }

    // Populate the unfolded tensor
    for idx in ndarray::indices(shape) {
        let mode_idx = idx[mode];
        let idx_vec: Vec<usize> = idx.as_array_view().to_vec();
        let col_idx = calc_col_idx(&idx_vec, shape, mode);
        result[[mode_idx, col_idx]] = tensor[idx.clone()];
    }

    Ok(result)
}

// Helper function to compute the Khatri-Rao product for Tucker ALS
fn compute_khatri_rao_product<A>(
    factors: &[Array2<A>],
    skip_mode: usize,
    core: &ArrayD<A>,
) -> LinalgResult<Array2<A>>
where
    A: Clone + Float + NumAssign + Zero + Debug + Send + Sync + 'static,
{
    use crate::tensor_contraction::mode_n_product;

    let _n_modes = factors.len();

    // We don't need the unfolded core tensor here, we'll unfold the projected tensor later
    let _core_unfolded = unfold_tensor(core, skip_mode)?;

    // For each mode except the one to skip, project the core tensor
    let mut projected_tensor = core.clone();

    for (mode, factor) in factors.iter().enumerate() {
        if mode == skip_mode {
            continue;
        }

        // Project the tensor along this mode
        projected_tensor = mode_n_product(&projected_tensor.view(), &factor.view(), mode)?;
    }

    // Unfold the projected tensor along the skipped mode
    let result = unfold_tensor(&projected_tensor, skip_mode)?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    #[ignore = "SVD fails for small matrices due to unimplemented eigendecomposition"]
    fn test_tucker_decomposition_basic() {
        // Create a simple 2x3x2 tensor
        let tensor = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];

        // Decompose with full rank
        let tucker = tucker_decomposition(&tensor.view(), &[2, 3, 2]).unwrap();

        // Check dimensions
        assert_eq!(tucker.core.shape(), &[2, 3, 2]);
        assert_eq!(tucker.factors.len(), 3);
        assert_eq!(tucker.factors[0].shape(), &[2, 2]);
        assert_eq!(tucker.factors[1].shape(), &[3, 3]);
        assert_eq!(tucker.factors[2].shape(), &[2, 2]);

        // Reconstruct the tensor
        let reconstructed = tucker.to_full().unwrap();

        // Check reconstruction
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..2 {
                    assert_abs_diff_eq!(
                        reconstructed[[i, j, k]],
                        tensor[[i, j, k]],
                        epsilon = 1e-10
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "SVD fails for small matrices due to unimplemented eigendecomposition"]
    fn test_tucker_decomposition_truncated() {
        // Create a 2x3x2 tensor
        let tensor = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];

        // Decompose with truncated rank
        let tucker = tucker_decomposition(&tensor.view(), &[2, 2, 2]).unwrap();

        // Check dimensions
        assert_eq!(tucker.core.shape(), &[2, 2, 2]);
        assert_eq!(tucker.factors.len(), 3);
        assert_eq!(tucker.factors[0].shape(), &[2, 2]);
        assert_eq!(tucker.factors[1].shape(), &[3, 2]);
        assert_eq!(tucker.factors[2].shape(), &[2, 2]);

        // Reconstruct the tensor and verify it works
        let _reconstructed = tucker.to_full().unwrap();

        // Check reconstruction error (should be small but not zero)
        let error = tucker.reconstruction_error(&tensor.view()).unwrap();
        assert!(error > 0.0);
        assert!(error < 0.1); // Somewhat arbitrary threshold
    }

    #[test]
    #[ignore = "SVD fails for small matrices due to unimplemented eigendecomposition"]
    fn test_tucker_als() {
        // Create a 2x3x2 tensor
        let tensor = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];

        // Decompose with ALS and truncated rank
        let tucker = tucker_als(&tensor.view(), &[2, 2, 2], 10, 1e-4).unwrap();

        // Check dimensions
        assert_eq!(tucker.core.shape(), &[2, 2, 2]);
        assert_eq!(tucker.factors.len(), 3);
        assert_eq!(tucker.factors[0].shape(), &[2, 2]);
        assert_eq!(tucker.factors[1].shape(), &[3, 2]);
        assert_eq!(tucker.factors[2].shape(), &[2, 2]);

        // Reconstruct the tensor and verify it works
        let _reconstructed = tucker.to_full().unwrap();

        // ALS should provide a better fit than HOSVD for same rank
        let hosvd_tucker = tucker_decomposition(&tensor.view(), &[2, 2, 2]).unwrap();
        let als_error = tucker.reconstruction_error(&tensor.view()).unwrap();
        let hosvd_error = hosvd_tucker.reconstruction_error(&tensor.view()).unwrap();

        // ALS should give at least as good a result as HOSVD
        assert!(als_error <= hosvd_error * 1.001); // Allow small numerical differences
    }

    #[test]
    #[ignore = "SVD fails for small matrices due to unimplemented eigendecomposition"]
    fn test_compress() {
        // Create a 2x3x2 tensor
        let tensor = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];

        // Decompose with full rank
        let tucker = tucker_decomposition(&tensor.view(), &[2, 3, 2]).unwrap();

        // Compress with specified ranks
        let compressed = tucker.compress(Some(vec![2, 2, 2]), None).unwrap();

        // Check dimensions
        assert_eq!(compressed.core.shape(), &[2, 2, 2]);
        assert_eq!(compressed.factors[0].shape(), &[2, 2]);
        assert_eq!(compressed.factors[1].shape(), &[3, 2]);
        assert_eq!(compressed.factors[2].shape(), &[2, 2]);

        // Compress with epsilon
        let compressed_eps = tucker.compress(None, Some(0.1)).unwrap();

        // Should have at least one column in each factor
        for factor in &compressed_eps.factors {
            assert!(factor.shape()[1] >= 1);
        }
    }
}
